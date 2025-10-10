"""Logging configuration with OpenTelemetry support"""

import logging
import sys
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor


def setup_logging(level: str = "INFO", format_json: bool = False) -> None:
    """Setup logging configuration

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_json: Whether to use JSON formatting
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    if format_json:
        # For production, use structured JSON logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        # For development, use human-readable format with source location
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set specific log levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def setup_tracing(
    service_name: str = "mlagentfactory",
    export_to_console: bool = True,
    jaeger_endpoint: Optional[str] = None
) -> trace.Tracer:
    """Setup OpenTelemetry tracing

    Args:
        service_name: Name of the service for tracing
        export_to_console: Whether to export traces to console
        jaeger_endpoint: Optional Jaeger endpoint for trace export

    Returns:
        Configured tracer instance
    """
    # Create resource with service information
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "0.1.0"
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add console exporter if requested
    if export_to_console:
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(console_exporter)
        provider.add_span_processor(console_processor)

    # Add Jaeger exporter if endpoint provided
    if jaeger_endpoint:
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            jaeger_processor = BatchSpanProcessor(jaeger_exporter)
            provider.add_span_processor(jaeger_processor)
        except ImportError:
            logging.warning("Jaeger exporter not available. Install with: pip install opentelemetry-exporter-jaeger")

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Instrument HTTP requests
    RequestsInstrumentor().instrument()

    # Get tracer
    tracer = trace.get_tracer(__name__)

    logging.info(f"Tracing initialized for service: {service_name}")

    return tracer


def get_tracer(name: str = __name__) -> trace.Tracer:
    """Get a tracer instance

    Args:
        name: Name for the tracer

    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


class TracedLogger:
    """Logger wrapper that adds trace context to log messages"""

    def __init__(self, name: str):
        """Initialize traced logger

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)

    def _get_trace_context(self) -> str:
        """Get current trace context as string

        Returns:
            Trace context string
        """
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            ctx = span.get_span_context()
            return f"[trace_id={format(ctx.trace_id, '032x')} span_id={format(ctx.span_id, '016x')}]"
        return ""

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message with trace context"""
        trace_ctx = self._get_trace_context()
        self.logger.debug(f"{trace_ctx} {msg}", *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message with trace context"""
        trace_ctx = self._get_trace_context()
        self.logger.info(f"{trace_ctx} {msg}", *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message with trace context"""
        trace_ctx = self._get_trace_context()
        self.logger.warning(f"{trace_ctx} {msg}", *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message with trace context"""
        trace_ctx = self._get_trace_context()
        self.logger.error(f"{trace_ctx} {msg}", *args, **kwargs)


# Convenience function to initialize everything
def initialize_observability(
    log_level: str = "INFO",
    service_name: str = "mlagentfactory",
    enable_tracing: bool = True,
    export_traces_to_console: bool = False
) -> Optional[trace.Tracer]:
    """Initialize logging and tracing

    Args:
        log_level: Logging level
        service_name: Service name for tracing
        enable_tracing: Whether to enable tracing
        export_traces_to_console: Whether to export traces to console

    Returns:
        Tracer instance if tracing enabled, None otherwise
    """
    # Setup logging
    setup_logging(level=log_level)

    # Setup tracing if enabled
    tracer = None
    if enable_tracing:
        tracer = setup_tracing(
            service_name=service_name,
            export_to_console=export_traces_to_console
        )

    return tracer
