def test_callable_event_sink_forwards_event_to_callback() -> None:
    """Fails if the CLI adapter stops delivering an event to its callback."""
    from fall_detection.adapters.event_sink import CallableEventSink

    delivered = []
    sink = CallableEventSink(delivered.append)

    sink.publish("detected")

    assert delivered == ["detected"]
