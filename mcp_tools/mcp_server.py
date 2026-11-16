from typing import Annotated

import os

from fastmcp import FastMCP
from pydantic import Field

from db.postgrest_client import fetch
from schemas.records import Customer, Order
from schemas.tool_result import ToolResult, MAX_LIMIT

mcp = FastMCP("ABEG Ops PostgREST Server")

# Reusable argument types: the description is what the model reads for each argument.
# Limit = Annotated[int, Field(description=f"Maximum rows to return. At most {MAX_LIMIT}; larger values are capped.")]
# Offset = Annotated[int, Field(description="Rows to skip, for paging. Use meta.has_more to know if there is a next page.")]
CustomerRef = Annotated[str, Field(description="The customer ID exactly as it appears in the conversation (e.g. CUSTOMER_1). Pass it unchanged.")]
OrderRef = Annotated[str, Field(description="The order ID exactly as it appears in the conversation. Pass it unchanged.")]

@mcp.tool()
async def get_all_customers(limit: int = 5, offset: int = 0) -> ToolResult[Customer]:
    """List customers with their contact details, a page at a time, ordered by customer ID."""
    return await fetch(tool="get_all+customers", model=Customer, table="customers", filters={"order": "customer_id.asc"}, limit=limit, offset=offset)

@mcp.tool()
async def get_customer_by_id(customer_id: CustomerRef, limit: int = 5, offset: int = 0) -> ToolResult[Customer]:
    """Get a customer by their ID."""
    return await fetch(tool="get_customer_by_id", model=Customer, table="customers", filters={"customer_id": f"eq.{customer_id}"}, limit=limit, offset=offset)

@mcp.tool()
async def get_all_orders(limit: int = 5, offset: int = 0) -> ToolResult[Order]:
    """List orders with their details, a page at a time, ordered by order ID."""
    return await fetch(tool="get_all_orders", model=Order, table="orders", filters={"order": "order_id.asc"}, limit=limit, offset=offset)

@mcp.tool()
async def get_order_by_customer_id(customer_id: CustomerRef, limit: int = 5, offset: int = 0) -> ToolResult[Order]:
    """Get orders associated with a specific customer."""
    return await fetch(tool="get_order_by_customer_id", model=Order, table="orders", filters={"customer_id": f"eq.{customer_id}"}, limit=limit, offset=offset)

@mcp.tool()
async def get_order_by_id(order_id: OrderRef, limit: int = 5, offset: int = 0) -> ToolResult[Order]:
    """Get a specific order by its ID."""
    return await fetch(tool="get_order_by_id", model=Order, table="orders", filters={"order_id": f"eq.{order_id}"}, limit=limit, offset=offset)

if __name__ == "__main__":
    print("Configured PostgREST ABEG Ops PostgREST Server is running.")
    mcp.run(
        transport="http",
        host=os.getenv("MCP_HOST","127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "9000")),
        show_banner=False
    )
