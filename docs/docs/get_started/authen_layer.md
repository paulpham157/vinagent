# Authentication

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent-docs/blob/main/docs/tutorials/get_started/authen_layer.ipynb)

Vinagent ensures AI Agent security through OAuth 2.0 authentication, a protocol that allows third-party applications to access Agent resources without exposing any user's credentials. This approach uses access token instread of direct user/password authentication. It works by orchestrating these participants.

## OAuth2 Architecture

The authentication system involves four key participants:

- Client (Business Client): The application that wants to work with the AI Agent.
- Authorization Server (OAuth Server): Issues tokens after verifying identity and permissions.
- Resource Server (AI Agent): Hosts the protected resource, here is Agent inference ability.
- User: The owner of the resource who grants permission.

![](../images/OAuth2.png)

## Setup OAuth2

First, create fake user credentials for demonstration purposes:

```python
%cd vinagent/vinagent/oauth2
!python3 user_gen.py --save_path authen/secret.json
```

This command creates a test user profile stored in authen/secret.json, simulating data that would typically exist in a production database.

Examine the generated user data:

```python
!cat authen/secret.json
```

    {"secret_key": "171d7a898dfcd817742364fac151dfce7328f0c88b720909279627ec5cd93197", "username": "Kan", "password": "password123", "hashed_password": "$2b$12$qGDJKEn.86b7Ol21M2J3fOG0BNKVXmQYpssdImOI73ZV.t7PEPwE2", "algorithm": "HS256", "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJoYXNoZWRfcGFzc3dvcmQiOiIkMmIkMTIkcUdESktFbi44NmI3T2wyMU0ySjNmT0cwQk5LVlhtUVlwc3NkSW1PSTczWlYudDdQRVB3RTIiLCJleHAiOjE3NTMyMDQ3MzksImlhdCI6MTc1MzIwMTEzOX0.OLnzrupahZGyi3d4C3LdDhpaTuaW1_mCMxl4e91Li0s", "api_url": "http://localhost:8000/verify-token"}

## OAuth2 Server

Launch the FastAPI authentication server. Let's ensure you are at `vinagent/vinagent/oauth2` directory before running the server. You should run server on terminal:

```python
!python3 server.py
```

    INFO:     Started server process [58893]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)


## Vinagent security layer

Here's how to implement authentication in your Vinagent application:

```python
from langchain_together import ChatTogether 
from vinagent.agent.agent import Agent
from vinagent.oauth2.client import AuthenCard
from dotenv import load_dotenv
load_dotenv()

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

# Step 1: Create AuthenCard to verify user token
authen_card = AuthenCard.from_config("authen/secret.json")

# Step 2: Create Agent with authen_card
agent = Agent(
    description="You are a Financial Analyst",
    llm = llm,
    skills = [
        "Deeply analyzing financial markets", 
        "Searching information about stock price",
        "Visualization about stock price"
    ],
    authen_card = authen_card
)

# Step 3: invoke the agent
message = agent.invoke("Who you are?")
print(message)
```

    INFO:vinagent.agent.agent:Successfully authenticated!
    INFO:vinagent.agent.agent:I'am chatting with unknown_user
    INFO:httpx:HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"


    content='I am a Financial Analyst.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 295, 'total_tokens': 302, 'completion_tokens_details': None, 'prompt_tokens_details': None, 'cached_tokens': 0}, 'model_name': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-e13e16c8-2d63-4e54-87c7-af47f171f623-0' usage_metadata={'input_tokens': 295, 'output_tokens': 7, 'total_tokens': 302, 'input_token_details': {}, 'output_token_details': {}}


You can manually test token authentication:

```python
authen_card.verify_access_token(
    token="your_token_there",
    api_url="http://localhost:8000/verify-token"
)
```

    True

## Using FastAPI Swagger UI
For interactive testing and token generation:

- Navigate to the FastAPI Swagger UI
- Click "Authorize" and login with admin credentials
- Use the `/token` endpoint to generate new tokens with `username/password`


## Security Best Practices
To ensure secure authentication, you should consider the following security best practices. First, always store tokens securely and never expose them in client-side code, as this prevents unauthorized access to sensitive credentials. Additionally, implement token refresh mechanisms for long-running applications to maintain continuous authentication without requiring users to re-authenticate frequently. It's also crucial to use HTTPS in production environments to encrypt data transmission and protect against man-in-the-middle attacks. Furthermore, regularly rotate secret keys and tokens to minimize the risk of compromised credentials, and consistently monitor authentication logs for suspicious activity to detect potential security breaches early and respond accordingly.

## Troubleshooting

Common Issues:

- **Server not responding**: Ensure the OAuth server is running on the correct port
- **Token expired**: Generate a new token using the `/token` endpoint
- **Authentication failed**: Verify the token format and server URL are correct

For additional support, refer to the Vinagent documentation or check server logs for detailed error messages.
