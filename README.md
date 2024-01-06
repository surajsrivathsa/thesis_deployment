# thesis_deployment


# Comic Search Engine - Adaptive and Explainable

![image](https://github.com/surajsrivathsa/thesis_deployment/assets/15963895/a171b7ce-e41a-44e6-bca6-6e44d4e812cc)

![image](https://github.com/surajsrivathsa/thesis_deployment/assets/15963895/6df5f3be-423a-4f63-b6c7-100f050e29a2)


## Overview

The Comic Search Engine is a powerful and user-centered search system designed for comic book enthusiasts. It leverages adaptive personalization and explanations to provide an enhanced search experience. This project is based on research conducted for a master's thesis in Data Science.

### Key Features

- **Adaptive Personalization**: Our system dynamically tailors search results based on user behavior and preferences.
- **Explanations**: Users receive explanations for personalized results, enhancing transparency and trust.
- **Multi-Modal Search**: Combining textual analysis and visual cues for more accurate and diverse search results.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/surajsrivathsa/thesis_deployment.git
   ```

2. Navigate to the project directory:

   ```shell
   cd thesis_deployment
   ```

3. Set up the frontend and backend components following the respective instructions below.

Run Docker Compose to run both the frontend as well as backend 
else you need to run below components individually

```shell
docker-compose up -d
```

### Frontend

1. Navigate to the `frontend` directory:

   ```shell
   cd react_frontend_ui
   ```

2. Install dependencies:

   ```shell
   npm install
   ```

3. Start the frontend server:

   ```shell
   npm start
   ```

### Backend

1. Navigate to the `backend` directory:

   ```shell
   cd python_backend_api
   ```

2. Install dependencies:

   ```shell
   pip install -r requirements.txt
   ```

3. Start the backend server:

   ```shell
   cd ./fastapi_webserver
   python search_main.py
   ```

## Usage

1. Access the Comic Search Engine via your web browser.
2. Enter your search query, such as a comic book title or character.
3. Explore the personalized search results and explanations.
4. Enjoy discovering new comics tailored to your preferences!
5. Tailor your preference by chnaging slider weights and clicking back on the search book again

## Contributing

Contributions to this project are welcome. Feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).


Please replace images, `surajsrivathsa`, and customize any other placeholders with the relevant information for your GitHub repository. Additionally, you can further expand on the installation, usage, and contributing sections as needed for your specific project.
