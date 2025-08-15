# GenAI-Powered Mini-SOAR for Phishing Analysis

This project is a prototype Security Orchestration, Automation, and Response (SOAR) application built with Python. It uses a machine learning model to predict if a URL is malicious and leverages Generative AI to prescribe a response plan. The entire application is containerized with Docker and orchestrated with Docker Compose for easy setup and deployment.

## Features

-   **Predictive Analytics**: Uses PyCaret to automatically train a model on a real-world phishing dataset.
-   **Prescriptive Analytics**: Integrates with Google Gemini, OpenAI, and Grok to generate detailed incident response plans.
-   **Interactive UI**: A user-friendly web interface built with the latest version of Streamlit.
-   **Containerized**: Fully containerized with Docker and managed with Docker Compose for a reproducible environment.
-   **Simplified Workflow**: A `Makefile` provides simple commands for building, running, and managing the application.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
-   [Docker](https://www.docker.com/get-started)
-   [Docker Compose](https://docs.docker.com/compose/install/) (often included with Docker Desktop)
-   [Make](https://www.gnu.org/software/make/) (pre-installed on most Linux/macOS systems; Windows users can use WSL or Chocolatey).
-   API keys for at least one Generative AI service (Gemini, OpenAI, or Grok).

## Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd mini-soar
    ```

2.  **Configure API Keys**
    This is the most important step. The application reads API keys from a `secrets.toml` file.

    -   Create the directory and file:
        ```bash
        mkdir -p .streamlit
        touch .streamlit/secrets.toml
        ```
    -   Open `.streamlit/secrets.toml` and add your API keys. Use the following template:
        ```toml
        # .streamlit/secrets.toml
        OPENAI_API_KEY = "sk-..."
        GEMINI_API_KEY = "AIza..."
        GROK_API_KEY = "gsk_..."
        ```
        *You only need to provide a key for the service(s) you intend to use.*

## Running the Application

With the `Makefile`, running the application is simple.

-   **To build and start the application:**
    ```bash
    make up
    ```
    The first time you run this, it will download the necessary Docker images and build the application container. This may take a few minutes. Subsequent runs will be much faster.

-   Once it's running, open your web browser and go to:
    **[http://localhost:8501](http://localhost:8501)**

-   **To view the application logs:**
    ```bash
    make logs
    ```

-   **To stop the application:**
    ```bash
    make down
    ```

-   **To perform a full cleanup** (stops containers and removes generated model/data files):
    ```bash
    make clean
    ```

## Project Structure
```
mini-soar/
├── README.md
├── Makefile
├── app.py
├── train_model.py
├── genai_prescriptions.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .streamlit/
    └── secrets.toml
```
