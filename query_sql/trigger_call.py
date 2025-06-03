import os
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve Twilio credentials from environment variables
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")

# Validate environment variables
if not account_sid or not auth_token:
    logger.error("TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN is not set in the .env file")
    raise ValueError("Missing Twilio credentials in .env file")

# Initialize Twilio client
try:
    client = Client(account_sid, auth_token)
    logger.info("Twilio client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Twilio client: {str(e)}")
    raise

# Trigger an outbound call
try:
    call = client.calls.create(
        to="+919786057537",  # Your Indian phone number
        from_="+18312152868",  # Your Twilio number
        url="https://d65d-2409-40f4-101c-c09-8cf6-8a8f-d142-9955.ngrok-free.app/voice",  # Webhook for TwiML
        method="POST"
    )
    logger.info(f"Call initiated successfully: SID={call.sid}")
    print(f"Call initiated: {call.sid}")
except TwilioRestException as e:
    logger.error(f"Twilio API error: {str(e)}")
    print(f"Error initiating call: {str(e)}")
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    print(f"Unexpected error: {str(e)}")