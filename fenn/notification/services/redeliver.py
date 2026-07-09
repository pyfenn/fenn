import resend

from fenn.notification.service import Service


class Resend(Service):
    """Resend email notification service."""

    def __init__(
        self,
        api_key: str,
        from_email: str,
        to_emails_raw: str,
        subject: str = "Notification from fenn",
    ):
        """Initialize Resend service.

        Args:
            subject: Email subject line. Defaults to "Notification from fenn".

        Raises:
            KeyError: If required configuration is missing.
        """
        super().__init__()

        self._api_key = api_key
        self._from_email = from_email
        self._to_emails_raw = to_emails_raw

        self._to_emails = [email.strip() for email in self._to_emails_raw.split(",")]

        self._subject = subject

        resend.api_key = self._api_key

    def send_notification(self, message: str) -> None:
        """Send email notification to all configured recipients.

        Args:
            message: The message to send as email body.

        Raises:
            RuntimeError: If the email fails to send.
        """
        try:
            params = {
                "from": self._from_email,
                "to": self._to_emails,
                "subject": self._subject,
                "html": f"<p>{message}</p>",
            }

            response = resend.Emails.send(params)  # ty: ignore[invalid-argument-type]

            if isinstance(response, dict) and "error" in response:
                raise RuntimeError(f"Resend API error: {response['error']}")

        except Exception as err:
            raise RuntimeError(f"Failed to send email notification: {err}") from err
