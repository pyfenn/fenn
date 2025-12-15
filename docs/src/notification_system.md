# PyFenn Notification System

A flexible and extensible notification system for ML training alerts that supports services like Discord and Telegram

## Configuration

### Environment Variables

Add these environment variables to `.env` to enable different notification services:

#### Discord

To get a Discord webhook URL, you need to create it in a server where you have permission (typically “Manage Webhooks”) and then copy its generated URL for later use.

## Create the webhook

1. Open Discord and go to the server where you want to receive messages.
2. Click the server name (top-left) and choose “Server Settings”.
3. In the left sidebar, go to **Integrations** and **Webhooks**.
4. Click **New Webhook** / **Create Webhook**.

## Copy the webhook URL

1. In the same Webhook screen, click **Copy Webhook URL**.
2. The copied string will look like:
   ```text
   https://discord.com/api/webhooks/XXXXXXXX/XXXXXXXXXXXXXXXX
   ```
   where the first part identifies the webhook and the second is its token.

Now you can add these key to your `.env` file as shown below:

```bash
DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"
```

#### Telegram

By default Telegram needs two values: a TELEGRAM_BOT_TOKEN and a TELEGRAM_CHAT_ID.
A Telegram bot token is obtained from BotFather when you create a bot, and the chat ID is read either from `getUpdates`.

## Get the bot token

1. Open Telegram and search for `@BotFather`.
2. Start the chat and send `/newbot`.
3. Follow the prompts to set:
   - A display name.
   - A unique username ending with `bot` (e.g. `my_test_bot`).
4. At the end, BotFather will show a line like:
   `Use this token to access the HTTP API: 123456:ABC-DEF...`
5. Copy that token and store it safely; it is your `YOUR_BOT_TOKEN`.

## Get your chat ID

1. Start a chat with your bot (search its username, press “Start”) and send any message (e.g. `hi`).
2. In a browser, open:
   ```text
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
   replacing `<YOUR_BOT_TOKEN>` with the token from BotFather.

3. In the JSON response, find:
   ```json
   "chat": {
     "id": 123456789,
     ...
   }
   ```
   The numeric value in `id` is `YOUR_CHAT_ID`.

Now you can add those key to your `.env` file as shown below:

```bash
TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID="YOUR_CHAT_ID"
```

#### Resend (Email)

Resend is a modern email API for developers. To use email notifications, you'll need a Resend API key and verified sender domain.

## Get your API key

1. Sign up at [resend.com](https://resend.com)
2. Go to the [API Keys](https://resend.com/api-keys) page in your dashboard
3. Click **Create API Key**
4. Give it a name (e.g., "fenn-notifications") and select appropriate permissions
5. Copy the API key (it starts with `re_`)
   > **Important**: Save this key securely - you won't be able to see it again!

## Verify your domain (optional but recommended)

For production use, verify your own domain:
1. Go to [Domains](https://resend.com/domains) in your dashboard
2. Click **Add Domain** and follow the DNS setup instructions
3. Once verified, you can send from any address at that domain (e.g., `notifications@yourdomain.com`)

For testing, you can use Resend's test domain: `onboarding@resend.dev`

## Configure environment variables

Now you can add these keys to your `.env` file as shown below:

```bash
RESEND_API_KEY="re_your_api_key_here"
RESEND_FROM_EMAIL="onboarding@resend.dev"  # or your verified domain email
RESEND_TO_EMAILS="recipient1@example.com,recipient2@example.com"  # comma-separated list
```

#### Slack

Slack notifications use incoming webhooks to send messages to a specific channel. You'll need to create a webhook URL for your workspace.

## Create the webhook

1. Open your Slack workspace in a browser
2. Go to [Slack API: Incoming Webhooks](https://api.slack.com/messaging/webhooks)
3. Click **Create your Slack app** (if you don't have one already)
4. Choose **From scratch**
5. Give your app a name (e.g., "Fenn Notifications") and select your workspace
6. In the app settings, go to **Incoming Webhooks** in the left sidebar
7. Toggle **Activate Incoming Webhooks** to **On**
8. Click **Add New Webhook to Workspace**
9. Select the channel where you want to receive notifications
10. Click **Allow**

## Copy the webhook URL

1. After authorization, you'll see your webhook URL in the **Webhook URLs for Your Workspace** section
2. The URL will look like:
   ```text
   https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX
   ```
3. Click **Copy** to copy the webhook URL

Now you can add this key to your `.env` file as shown below:

```bash
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

### Basic Usage

```python
from fenn.notification import Notifier
from fenn.notification.services import Discord, Telegram, Resend, Slack

# Create notifier
notifier = Notifier()

# Add services
notifier.add_service(Discord)   # Requires DISCORD_WEBHOOK_URL env var
notifier.add_service(Telegram)  # Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars
notifier.add_service(Resend)    # Requires RESEND_API_KEY, RESEND_FROM_EMAIL, and RESEND_TO_EMAILS env vars
notifier.add_service(Slack)     # Requires SLACK_WEBHOOK_URL env var

# or you can just notifier.add_services([Discord, Telegram, Resend, Slack])

# Send notification
notifier.notify("Hello from PyFenn!")
```

### Using Email with Custom Subject

```python
from fenn.notification import Notifier
from fenn.notification.services import Resend

notifier = Notifier()

# Create Resend service with custom subject
resend_service = Resend(subject="Training Alert from ML Pipeline")

# Add the service instance to notifier
notifier._services.append(resend_service)

# Send notification
notifier.notify("Model training completed successfully!")
```


### Error Handling

The notification system handles errors gracefully:

```python
notifier = Notifier()
notifier.add_service(Discord)  # This might fail
notifier.add_service(Slack)    # This might work

# Even if some services fail, others will still work
notifier.notify("This message will be sent to working services only")
```

### Service Management

```python
from fenn.notification import Notifier
from fenn.notification.services import Discord, Telegram

notifier = Notifier()

# Add service
notifier.add_services([Discord, Telegram])

# Check registered services
print(notifier.get_services())  # ['Discord', 'Telegram']

# Remove a service
notifier.remove_service(Discord)
print(notifier.get_services())  # ['Telegram']

# Clear all services
notifier.clear_services()
print(notifier.get_services())  # []
```

## Creating Custom Services

You can easily create custom notification services by implementing the `Service` interface:

```python
from fenn.notification import Notifier, Service
import requests

class CustomWebhook(Service):

    def __init__(self):
        super().__init__()
        self._webhook_url = self._keystore.get_key("WEBHOOK_URL")

    def send_notification(self, message: str) -> None:
        try:
            response = requests.post(
            self._webhook_url,
            json={"text": message},
            timeout=10
        )
        response.raise_for_status()

        except requests.exceptions.RequestException as err:
            raise requests.exceptions.RequestException(f"Failed to send notification: {err}")

# Use your custom service
notifier = Notifier()
notifier.add_service(CustomWebhook)
notifier.notify("Hello from custom service!")
```