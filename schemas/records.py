from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import Field, BaseModel, ConfigDict


class Sensitivity(str, Enum):
    IDENTIFIER = "identifier"
    PII = "pii"
    NONE = "none"

def field(sensitivity: Sensitivity = Sensitivity.NONE, **kwargs):
    return Field(json_schema_extra={"x-sensitivity": sensitivity.value}, **kwargs)

class Record(BaseModel):
    model_config = ConfigDict(extra="ignore")

class CustomerContact(Record):
    customer_phone_number: str | None = field(Sensitivity.PII, default=None)
    customer_alt_phone_number: str | None = field(Sensitivity.PII, default=None)
    customer_email: str | None = field(Sensitivity.PII, default=None)
    customer_date_of_birth: date | None = field(Sensitivity.PII, default=None)


class Customer(Record):
    customer_id: str = field(Sensitivity.IDENTIFIER)
    customer_first_name: str = field(Sensitivity.PII)
    customer_last_name: str | None = field(Sensitivity.PII, default=None)
    customer_status: Literal["ACTIVE", "INACTIVE", "BLOCKED"] = field()
    customer_type: Literal["Retail", "Media", "Ecomm"] | None = field(default=None)
    customer_joining_date: date = field()
    customer_pii: list[CustomerContact] = field(default_factory=list)   # embedded from customer_pii


class Order(Record):
    order_id: str = field(Sensitivity.IDENTIFIER)
    customer_id: str = field(Sensitivity.IDENTIFIER)
    order_type: Literal["STANDARD", "RETURN", "ADHOC"] = field()
    original_order_id: str | None = field(Sensitivity.IDENTIFIER, default=None)
    order_channel: Literal["WEBSITE", "INSTAGRAM", "WHATSAPP", "FACEBOOK", "RETAIL", "STALL"] = field()
    order_status: Literal["PLACED", "CONFIRMED", "PACKED", "DISPATCHED", "OUT_FOR_DELIVERY", "DELIVERED",
    "FULFILLED", "CANCELLED", "RETURN_INITIATED", "REPLACEMENT_INITIATED"] = field()
    order_date_time: datetime = field()
    order_coupon_code: str | None = field(default=None)
    order_coupon_discount: Decimal = field()
    order_total_amount: Decimal | None = field(default=None)
    order_total_amount_currency: Literal["INR"] = field()
    contain_documents: bool = field()

def select_for(model: type[BaseModel]) -> str:
    """Build PostgREST's ?select= from a model, so we fetch exactly the model's fields.
    A list-of-Record field becomes an embed, e.g. customer_pii(customer_phone_number,...)."""
    parts = []
    for name, info in model.model_fields.items():
        args = getattr(info.annotation, "__args__", ())
        inner = args[0] if args else None
        if isinstance(inner, type) and issubclass(inner, Record):
            parts.append(f"{name}({select_for(inner)})")
        else:
            parts.append(name)
    return ",".join(parts)


def sensitive_fields(model: type[BaseModel]) -> dict[str, str]:
    """{field_name: sensitivity} for every tagged field, including embedded records."""
    out = {}
    for name, info in model.model_fields.items():
        tag = (info.json_schema_extra or {}).get("x-sensitivity", "none")
        if tag != "none":
            out[name] = tag
        args = getattr(info.annotation, "__args__", ())
        inner = args[0] if args else None
        if isinstance(inner, type) and issubclass(inner, Record):
            out.update(sensitive_fields(inner))
    return out








