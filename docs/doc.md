# installing deepface

## Emi trouble shooting notes to install deepface

when installing deepface on python 3.13.x I run into compatibility issues, so i installed pyenc to run this project pn python 3.11.9

## installation steps

1. install pyenv

```bash
brew install pyenv
```

2. update zsh terminal

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

3. reload variables for zsh terminal

```bash
source ~/.zshrc
```

4. set local python

```bash
pyenv local 3.11.9
```

5. verify correct python has been set

```bash
python3 --version
```

6. start clean virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptool
pip install deepface
```

7. pip install deepface

```bash
pip install deepface
```
