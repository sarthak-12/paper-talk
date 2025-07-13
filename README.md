# Paper-Talk: An AI-Powered Q&A Bot 🤖

Paper-talk is a web service that uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions about a specific document. This project demonstrates a complete MLOps workflow, from data processing and containerization to automated deployment.

### About The Project

The goal of this project is to provide a simple API that can ingest a complex document (in this case, the "Attention Is All You Need" research paper) and allow users to ask natural language questions about its content. The system finds the most relevant passages from the text and uses a Large Language Model (LLM) to generate a coherent, accurate answer.

This repository serves as a practical example of operationalizing an NLP model using modern MLOps principles.

---

### Built With

This project leverages several key technologies:

* **Python:** The primary programming language for the application logic.
* **FastAPI:** A high-performance web framework for building the API.
* **Sentence-Transformers:** For generating high-quality semantic embeddings of the text.
* **Groq API (Llama 3):** For the generative AI component that formulates answers.
* **Docker:** For containerizing the application to ensure consistent environments.
* **GitHub Actions:** For Continuous Integration and Continuous Delivery (CI/CD).
* **Terraform & AWS:** For defining and deploying the cloud infrastructure (ECR, App Runner).

---

### Getting Started

To get a local copy up and running, follow these steps.

#### Prerequisites

* Python 3.9+
* Docker Desktop installed and running.
* A Groq API Key.

#### Local Installation

1.  **Clone the repo:**
    ```sh
    git clone [https://github.com/your-username/paper-talk.git](https://github.com/your-username/paper-talk.git)
    cd paper-talk
    ```
2.  **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
4.  **Create a `.env` file** in the root directory and add your Groq API key:
    ```
    GROQ_API_KEY="gsk_YourApiKeyHere"
    ```
5.  **Process the data:**
    (You would add instructions here if you want others to be able to re-process the paper)
    ```sh
    # python process_paper.py
    ```

---

### Usage

You can run the application locally using Docker.

1.  **Build the Docker image:**
    ```sh
    docker build -t paper-talk .
    ```
2.  **Run the container:**
    ```sh
    docker run --rm -p 8000:8000 -e GROQ_API_KEY="gsk_YourApiKeyHere" --name paper-talk-container paper-talk
    ```
3.  Access the API documentation at `http://127.0.0.1:8000/docs`.

---

### CI/CD Pipeline

This project is configured with a GitHub Actions workflow that automates the following on every push to the `main` branch:
1.  **Builds** the Docker image.
2.  **Pushes** the image to a container registry.
3.  **(Upcoming)** Deploys the new image to a cloud environment.
