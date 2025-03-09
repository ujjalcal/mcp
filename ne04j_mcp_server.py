import os
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from neo4j import GraphDatabase, Driver
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Initialize FastAPI
app = FastAPI(title="Neo4j MCP Server", 
              description="Model-Content-Protocol server for Neo4j databases")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neo4j driver connection
def get_db() -> Driver:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        # Test connection
        driver.verify_connectivity()
        return driver
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# Models
class NodeLabel(BaseModel):
    label: str
    count: int
    properties: List[str]

class RelationshipType(BaseModel):
    type: str
    count: int
    properties: List[str]
    source_labels: List[str]
    target_labels: List[str]

class DatabaseSchema(BaseModel):
    nodes: List[NodeLabel]
    relationships: List[RelationshipType]

class QueryRequest(BaseModel):
    cypher: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class QueryResult(BaseModel):
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class PromptTemplate(BaseModel):
    name: str
    description: str
    prompt: str
    example_parameters: Dict[str, Any] = Field(default_factory=dict)

# Schema extraction functions
def get_node_labels(driver):
    with driver.session() as session:
        result = session.run("""
        CALL apoc.meta.nodeTypeProperties()
        YIELD nodeType, nodeLabels, propertyName
        WITH nodeLabels, collect(propertyName) AS properties
        MATCH (n) WHERE ALL(label IN nodeLabels WHERE label IN labels(n))
        WITH nodeLabels, properties, count(n) AS nodeCount
        RETURN nodeLabels, properties, nodeCount
        ORDER BY nodeCount DESC
        """)
        
        node_labels = []
        for record in result:
            label = record["nodeLabels"][0] if record["nodeLabels"] else "Unknown"
            node_labels.append(NodeLabel(
                label=label,
                count=record["nodeCount"],
                properties=record["properties"]
            ))
        return node_labels

def get_relationship_types(driver):
    with driver.session() as session:
        result = session.run("""
        CALL apoc.meta.relTypeProperties()
        YIELD relType, sourceNodeLabels, targetNodeLabels, propertyName
        WITH relType, sourceNodeLabels, targetNodeLabels, collect(propertyName) AS properties
        MATCH ()-[r]->() WHERE type(r) = relType
        WITH relType, sourceNodeLabels, targetNodeLabels, properties, count(r) AS relCount
        RETURN relType, sourceNodeLabels, targetNodeLabels, properties, relCount
        ORDER BY relCount DESC
        """)
        
        rel_types = []
        for record in result:
            rel_types.append(RelationshipType(
                type=record["relType"],
                count=record["relCount"],
                properties=record["properties"],
                source_labels=record["sourceNodeLabels"],
                target_labels=record["targetNodeLabels"]
            ))
        return rel_types

# Endpoints
@app.get("/schema", response_model=DatabaseSchema)
def get_schema(driver: Driver = Depends(get_db)):
    """
    Retrieve the complete database schema including node labels and relationship types
    """
    try:
        nodes = get_node_labels(driver)
        relationships = get_relationship_types(driver)
        return DatabaseSchema(nodes=nodes, relationships=relationships)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema retrieval failed: {str(e)}")

@app.post("/query", response_model=QueryResult)
def execute_query(query: QueryRequest, driver: Driver = Depends(get_db)):
    """
    Execute a read-only Cypher query against the database
    """
    # Ensure query is read-only
    lower_query = query.cypher.lower()
    if any(keyword in lower_query for keyword in ["create", "delete", "remove", "set", "merge"]):
        raise HTTPException(status_code=403, detail="Only read-only queries are allowed")
    
    try:
        with driver.session() as session:
            result = session.run(query.cypher, query.parameters)
            records = [record.data() for record in result]
            
            # Get query stats
            summary = result.consume()
            metadata = {
                "nodes_created": summary.counters.nodes_created,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
                "relationships_deleted": summary.counters.relationships_deleted,
                "properties_set": summary.counters.properties_set,
                "execution_time_ms": summary.result_available_after
            }
            
            return QueryResult(results=records, metadata=metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

# Analysis prompts
@app.get("/prompts", response_model=List[PromptTemplate])
def get_analysis_prompts():
    """
    Get a list of predefined prompt templates for common Neo4j data analysis tasks
    """
    prompts = [
        PromptTemplate(
            name="Relationship Analysis",
            description="Analyze relationships between two node types",
            prompt="""
            Given the Neo4j database with {node_type_1} and {node_type_2} nodes, 
            I want to understand the relationships between them.
            
            Please help me:
            1. Find the most common relationship types between these nodes
            2. Identify the distribution of relationship properties
            3. Discover any interesting patterns or outliers
            
            Sample Cypher query to start with:
            ```
            MATCH (a:{node_type_1})-[r]->(b:{node_type_2})
            RETURN type(r) AS relationship_type, count(r) AS count
            ORDER BY count DESC
            LIMIT 10
            ```
            """,
            example_parameters={"node_type_1": "Person", "node_type_2": "Movie"}
        ),
        PromptTemplate(
            name="Path Discovery",
            description="Find paths between nodes of interest",
            prompt="""
            I'm looking to understand how {start_node_label} nodes with property {start_node_property}="{start_node_value}" 
            connect to {end_node_label} nodes with property {end_node_property}="{end_node_value}".
            
            Please help me:
            1. Find all possible paths between these nodes
            2. Identify the shortest path
            3. Analyze what nodes and relationships appear most frequently in these paths
            
            Sample Cypher query to start with:
            ```
            MATCH path = (a:{start_node_label} {{
                {start_node_property}: "{start_node_value}"
            }})-[*1..{max_depth}]->(b:{end_node_label} {{
                {end_node_property}: "{end_node_value}"
            }})
            RETURN path LIMIT 10
            ```
            """,
            example_parameters={
                "start_node_label": "Person", 
                "start_node_property": "name",
                "start_node_value": "Tom Hanks",
                "end_node_label": "Person",
                "end_node_property": "name",
                "end_node_value": "Kevin Bacon",
                "max_depth": 4
            }
        ),
        PromptTemplate(
            name="Property Distribution",
            description="Analyze the distribution of property values",
            prompt="""
            I want to understand the distribution of {property_name} across {node_label} nodes.
            
            Please help me:
            1. Calculate basic statistics (min, max, avg, std)
            2. Identify the most common values and their frequencies
            3. Detect any outliers or unusual patterns
            
            Sample Cypher query to start with:
            ```
            MATCH (n:{node_label})
            WHERE n.{property_name} IS NOT NULL
            RETURN 
                min(n.{property_name}) AS min_value,
                max(n.{property_name}) AS max_value,
                avg(n.{property_name}) AS avg_value,
                stDev(n.{property_name}) AS std_value
            ```
            
            And for frequency distribution:
            ```
            MATCH (n:{node_label})
            WHERE n.{property_name} IS NOT NULL
            RETURN n.{property_name} AS value, count(n) AS frequency
            ORDER BY frequency DESC
            LIMIT 20
            ```
            """,
            example_parameters={"node_label": "Movie", "property_name": "runtime"}
        ),
        PromptTemplate(
            name="Community Detection",
            description="Detect communities or clusters in the graph",
            prompt="""
            I want to identify communities or clusters within the graph based on {relationship_type} relationships.
            
            Please help me:
            1. Apply graph algorithms to detect communities
            2. Analyze the size and composition of each community
            3. Identify central nodes within each community
            
            Sample Cypher query to start with (requires GDS library):
            ```
            CALL gds.graph.project(
                'community-graph',
                '*',
                '{relationship_type}'
            )
            YIELD graphName;
            
            CALL gds.louvain.stream('community-graph')
            YIELD nodeId, communityId
            WITH gds.util.asNode(nodeId) AS node, communityId
            RETURN communityId, collect(node.{label_property}) AS members, count(*) AS size
            ORDER BY size DESC
            LIMIT 10
            ```
            """,
            example_parameters={"relationship_type": "FRIENDS_WITH", "label_property": "name"}
        )
    ]
    return prompts

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)