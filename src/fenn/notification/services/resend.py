import resend

from fenn.notification.service import Service


class Resend(Service):
    """Resend email notification service."""

    def __init__(self, subject: str = "Notification from fenn"):
        """Initialize Resend service.

        Args:
            subject: Email subject line. Defaults to "Notification from fenn".

        Raises:
            KeyError: If required configuration is missing.
        """
        super().__init__()

        self._api_key = self._keystore.get_key("RESEND_API_KEY")
        self._from_email = self._keystore.get_key("RESEND_FROM_EMAIL")
        self._to_emails_raw = self._keystore.get_key("RESEND_TO_EMAILS")

        self._to_emails = [email.strip() for email in self._to_emails_raw.split(",")]

        self._subject = subject

        resend.api_key = self._api_key

    def send_notification(self, message: str) -> None:
        """Send email notification to all configured recipients.

        Args:
            message: The message to send as email body.

        Raises:
            Exception: If the email fails to send.
        """
        try:
            params = {
                "from": self._from_email,
                "to": self._to_emails,
                "subject": self._subject,
                "html": f"<p>{message}</p>",
            }

            response = resend.Emails.send(params)

            if isinstance(response, dict) and "error" in response:
                raise Exception(f"Resend API error: {response['error']}")

        except Exception as err:
            raise Exception(f"Failed to send email notification: {err}")
