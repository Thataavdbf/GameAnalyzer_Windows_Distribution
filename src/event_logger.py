import datetime

class EventLogger:
    def __init__(self):
        self.events = []

    def log_event(self, timestamp, event_type, description, metadata=None):
        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "description": description,
            "metadata": metadata if metadata is not None else {}
        }
        self.events.append(event)
        print(f"Logged event at {timestamp}: {event_type} - {description}")

    def get_events(self):
        return self.events

    def save_events_to_file(self, filename="events.json"):
        import json
        with open(filename, "w") as f:
            json.dump(self.events, f, indent=4)
        print(f"Events saved to {filename}")

    def clear_events(self):
        self.events = []
        print("Events cleared.")

if __name__ == '__main__':
    logger = EventLogger()
    logger.log_event(datetime.datetime.now().isoformat(), "detection", "Player detected", {"player_id": 1, "location": "(100, 200)"})
    logger.log_event(datetime.datetime.now().isoformat(), "action", "Shot fired", {"weapon": "Liberator"})
    logger.save_events_to_file("test_events.json")
    print(logger.get_events())
    logger.clear_events()
    print(logger.get_events())


