# Gitflow cheatsheet

Install GitFlow (Linux):

    sudo apt install gitflow

Initialize a project:

    git flow init
    git add .
    git commit -m "initialized gitflow"
    git push --all origin -u

Feature branch:

    git flow feature start feature_branch
    ...
    git flow feature finish feature_branch

Release:

    git flow release start 0.0.1
    ...
    git flow release finish '0.0.1'
    git checkout main
    git push --all origin
    git checkout dev

Hotfix:

    git flow hotfix start hotfix_branch
    ...
    git flow hotfix finish hotfix_branch