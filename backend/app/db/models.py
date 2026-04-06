from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, TypeDecorator
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import json

Base = declarative_base()

class JsonEncodedDict(TypeDecorator):
    """Enables JSON storage by encoding/decoding on the fly."""
    impl = Text

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return json.loads(value)

class SpotTrade(Base):
    __tablename__ = "trades"
    
    id = Column(String, primary_key=True)  # Using String to keep Mongo _id format
    symbol = Column(String, index=True)
    side = Column(String)
    entry_price = Column(Float, default=0.0)
    quantity = Column(Float, default=0.0)
    tp = Column(Float, default=0.0)
    sl = Column(Float, default=0.0)
    status = Column(String, index=True)
    orders = Column(JSON, default=[]) # SQLAlchemy 1.4+ JSON type works with SQLite
    orderListId = Column(Integer, nullable=True)
    entry_fees = Column(Float, default=0.0)
    fee_asset = Column(String, nullable=True)
    exit_price = Column(Float, default=0.0)
    exit_fees = Column(Float, default=0.0)
    exit_fee_asset = Column(String, nullable=True)
    close_time = Column(Float, nullable=True)
    timestamp = Column(Float, default=datetime.utcnow().timestamp)
    close_reason = Column(String, nullable=True)
    error_msg = Column(Text, nullable=True)

class LeadTrade(Base):
    __tablename__ = "lead_trades"
    
    id = Column(String, primary_key=True)
    symbol = Column(String, index=True)
    side = Column(String)
    entry_price = Column(Float, default=0.0)
    quantity = Column(Float, default=0.0)
    leverage = Column(Integer, default=10)
    tp = Column(Float, default=0.0)
    sl = Column(Float, default=0.0)
    status = Column(String, index=True)
    protection_orders = Column(JSON, default=[])
    entry_fees = Column(Float, default=0.0)
    entry_fee_asset = Column(String, default="USDT")
    exit_price = Column(Float, default=0.0)
    exit_fees = Column(Float, default=0.0)
    exit_fee_asset = Column(String, default="USDT")
    close_time = Column(Float, nullable=True)
    close_reason = Column(String, nullable=True)
    timestamp = Column(Float, default=datetime.utcnow().timestamp)
    error = Column(Text, nullable=True)

class AuditLog(Base):
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_str = Column(String)
    timestamp = Column(Float, index=True)
    method = Column(String)
    endpoint = Column(String)
    request = Column(JSON)
    response = Column(JSON)
    status = Column(Integer)

class EndpointAudit(Base):
    __tablename__ = "endpoint_audit"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String)
    unix_time = Column(Float, index=True)
    method = Column(String)
    url = Column(String)
    status_code = Column(Integer)
    process_time_ms = Column(Float)
    request_body = Column(JSON, nullable=True)
