import boto3
from pypdf import PdfReader
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Read and extract text from the PDF file.
reader = PdfReader("knowledgefiles/Packages.pdf")
text = ''.join(page.extract_text() for page in reader.pages)

# Set the model ID.
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# Start the conversation with the text from the PDF.
conversation = [
    {
        "role": "user",
        "content": text,
    }
]

try:
    # Send the initial message to the model.
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 2000, "temperature": 0},
        additionalModelRequestFields={"top_k": 250}
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"]
    # print("Claude's Response:\n", response_text)

    # Now you can ask questions based on the content.
    question = "What is the cost of beach villa  in VILLA NAUTICA PARADISE ISLAND RESORT & SPA  in INR?"

    # Add the question to the conversation.
    conversation.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Send the question to the model.
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 2000, "temperature": 0},
        additionalModelRequestFields={"top_k": 250}
    )

    # Extract and print the response to the question.
    response_text = response["output"]["message"]["content"]
    print("Claude's Answer:\n", response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)
