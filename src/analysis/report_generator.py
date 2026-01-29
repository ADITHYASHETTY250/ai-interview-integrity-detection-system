from pathlib import Path

class ReportGenerator:
    def __init__(self, config=None):
        self.config = config or {}

    def generate_offline_report_from_session_log(
        self,
        session_log_path,
        output_format="html",
    ):
        """
        Stub method.
        Dashboard already renders reports using Flask templates.
        This function exists only to avoid crashes.
        """
        if isinstance(session_log_path, (str, Path)):
            return str(session_log_path)

        return None
