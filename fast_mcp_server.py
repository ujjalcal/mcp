from mcp.server.fastmcp import FastMCP, Context
from neo4j import GraphDatabase
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.DEBUG)

# Initialize FastMCP server
mcp = FastMCP("Neo4j MCP Server")

# Neo4j connection details
NEO4J_URI = "neo4j+s://1e30f4c4.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "pDMkrbwg1L__-3BHh46r-MD9-z6Frm8wnR__ZzFiVmM"

# Neo4j driver connection
def get_db():
    logging.debug("Establishing Neo4j database connection")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

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

class QueryRequest(BaseModel):
    cypher: str
    parameters: Dict[str, Any] = {}

# Function to fetch node labels
def fetch_node_labels(session) -> List[NodeLabel]:
    logging.debug("Fetching node labels")
    result = session.run("""
    CALL apoc.meta.nodeTypeProperties()
    YIELD nodeType, nodeLabels, propertyName
    WITH nodeLabels, collect(propertyName) AS properties
    MATCH (n) WHERE ALL(label IN nodeLabels WHERE label IN labels(n))
    WITH nodeLabels, properties, count(n) AS nodeCount
    RETURN nodeLabels, properties, nodeCount
    ORDER BY nodeCount DESC
    """)
    
    return [NodeLabel(label=record["nodeLabels"][0] if record["nodeLabels"] else "Unknown",
                      count=record["nodeCount"],
                      properties=record["properties"]) for record in result]

# Function to fetch relationship types
def fetch_relationship_types(session) -> List[RelationshipType]:
    logging.debug("Fetching relationship types")
    result = session.run("""
    CALL apoc.meta.relTypeProperties()
    YIELD relType, sourceNodeLabels, targetNodeLabels, propertyName
    WITH relType, sourceNodeLabels, targetNodeLabels, collect(propertyName) AS properties
    MATCH ()-[r]->() WHERE type(r) = relType
    WITH relType, sourceNodeLabels, targetNodeLabels, properties, count(r) AS relCount
    RETURN relType, sourceNodeLabels, targetNodeLabels, properties, relCount
    ORDER BY relCount DESC
    """)
    
    return [RelationshipType(type=record["relType"],
                             count=record["relCount"],
                             properties=record["properties"],
                             source_labels=record["sourceNodeLabels"],
                             target_labels=record["targetNodeLabels"]) for record in result]

# Define a resource to get the database schema
@mcp.resource("schema://database")
def get_schema() -> Dict[str, Any]:
    logging.debug("get schemas...")
    driver = get_db()
    with driver.session() as session:
        nodes = fetch_node_labels(session)
        relationships = fetch_relationship_types(session)
        return {"nodes": nodes, "relationships": relationships}

# Define a tool to execute a query
@mcp.tool()
def execute_query(query: QueryRequest) -> Dict[str, Any]:
    logging.debug("execute query...")
    driver = get_db()
    with driver.session() as session:
        result = session.run(query.cypher, query.parameters)
        records = [record.data() for record in result]
        summary = result.consume()
        metadata = {
            "nodes_created": summary.counters.nodes_created,
            "nodes_deleted": summary.counters.nodes_deleted,
            "relationships_created": summary.counters.relationships_created,
            "relationships_deleted": summary.counters.relationships_deleted,
            "properties_set": summary.counters.properties_set,
            "execution_time_ms": summary.result_available_after
        }
        return {"results": records, "metadata": metadata}

# Define prompts for analysis
@mcp.prompt()
def relationship_analysis_prompt(node_type_1: str, node_type_2: str) -> str:
    logging.debug("relationship analysis prompt...")
    return f"""
    Given the Neo4j database with {node_type_1} and {node_type_2} nodes, 
    I want to understand the relationships between them.

    Please help me:
    1. Find the most common relationship types between these nodes
    2. Identify the distribution of relationship properties
    3. Discover any interesting patterns or outliers

    Sample Cypher query to start with:
    MATCH (a:{node_type_1})-[r]->(b:{node_type_2})
    RETURN type(r) AS relationship_type, count(r) AS count
    ORDER BY count DESC
    LIMIT 10
    """

@mcp.prompt()
def path_discovery_prompt(start_node_label: str, start_node_property: str, start_node_value: str, end_node_label: str, end_node_property: str, end_node_value: str, max_depth: int) -> str:
    logging.debug("path discovery prompt...")
    return f"""
    I'm looking to understand how {start_node_label} nodes with property {start_node_property}="{start_node_value}" 
    connect to {end_node_label} nodes with property {end_node_property}="{end_node_value}".

    Please help me:
    1. Find all possible paths between these nodes
    2. Identify the shortest path
    3. Analyze what nodes and relationships appear most frequently in these paths

    Sample Cypher query to start with:
    MATCH path = (a:{start_node_label} {{
        {start_node_property}: "{start_node_value}"
    }})-[*1..{max_depth}]->(b:{end_node_label} {{
        {end_node_property}: "{end_node_value}"
    }})
    RETURN path LIMIT 10
    """

# Run the MCP server
if __name__ == "__main__":
    logging.debug("Starting MCP server")
    mcp.run()
    # mcp.run(host="0.0.0.0")