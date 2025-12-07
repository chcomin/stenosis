echo "------------------------------------------------"
echo "Conda will create the installation plan. Check if everything looks good."
echo "In particular, check if the Python, Pytorch and CUDA versions are correct."
echo "------------------------------------------------"
echo
conda env create -f environment.yml -p ./.venv --dry-run

echo
echo "------------------------------------------------"
read -p "Do you want to proceed with the installation? (y/n) " -n 1 -r
echo    # move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi

echo
echo "------------------------------------------------"
echo "Creating local Conda environment and installing packages..."
echo "Ignore Conda's instruction to 'activate' the environment."
echo "------------------------------------------------"
conda env create -f environment.yml -p ./.venv

echo
echo "------------------------------------------------"
echo "Installing in editable mode..."
echo "------------------------------------------------"
conda run -p ./.venv uv pip install --no-build-isolation --no-deps -e .

# Set the environment name to avoid Conda printing the full path in the prompt
conda run -p ./.venv conda config --env --set env_prompt "(stenosis) "

echo
echo "------------------------------------------------"
echo "Setup complete!"
echo "To start working, run: conda activate ./.venv"