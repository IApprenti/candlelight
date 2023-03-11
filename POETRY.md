# Poetry Cheatsheet

Add a package

   poetry add package_name
   
Run a script:

    poetry run python your_script.py
    
Run tests:

    poetry run pytest

Open in sub shell:

    poetry shell

Activate virtual env manually:

    source $(poetry env info --path)/bin/activate
