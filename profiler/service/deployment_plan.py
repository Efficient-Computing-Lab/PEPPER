import yaml

def generate_docker_compose(output_file="docker-compose.yml"):
    """
    Generates a docker-compose.yml file with customizable services.
    """

    compose = {
        "version": "3.9",
        "services": {
            "model": {
                "build": ".",
                "container_name": "ai_app",
                "ports": ["5000:5000"],
                "volumes": [
                    "trained_model:/trained_model/"
                ],
            }
        },
        "volumes": {
            "trained_model": {}
        }
    }

    # Write YAML file
    with open(output_file, "w") as f:
        yaml.dump(compose, f, sort_keys=False)

    print(f"✅ Docker Compose file generated: {output_file}")


if __name__ == "__main__":
    generate_docker_compose()
