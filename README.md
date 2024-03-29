# PHYS-359-Winter-2023
![Project description](https://img.shields.io/badge/classwork-shared%20scripts%20-blue)

A repository for students taking PHYS 359 in Winter 2023 to share code used for numerical calculations

## Instructions on setting up a local repository

1. If you haven't already, install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) on your computer.

2. Once installed, open a terminal and configure git with your name and email. You should use the same name and email as your GitHub account.
    ```console
    git config --global user.name "Your Name"
    ```
    ```console
    git config --global user.email "Your Email"
    ```

3. To work with the repository, you need to clone the repository. To do this, open a terminal (in administrator mode in Windows) and navigate to the directory where you want the repository folder to be cloned to. It's recommended that this note be in your user directory, which contains your commonly used directories for your desktop, downloads, documents, etc. If you haven't used a terminal before, simply copy the address of the desired directory and use:
    ```console
    cd PATH/TO/DIRECTORY
    ```
    If your directory path includes spaces, enclose the path with quotation marks. Once you have navigated to this directory in your terminal, enter the following command:
    ```console
    git clone https://github.com/zhanjack822/UW-2022-First-Year-Physics-Coding.git
    ```
    The cloned repository on your computer is referred to as a "local repository" while the repository on GitHub is referred to as the "remote repository." Once you have successfully created a 
    
4. Create a branch from the master branch with your name as the branch name.
    ```console
    git checkout master
    ```
    ```console
    git checkout -b YOUR_NAME
    ```
    Note that `checkout` without the `-b` flag is used to switch between branches.

5. In the home directory of your local repository, create a folder to store your code. You can do this using the desktop GUI or in a terminal using `mkdir YOUR_NAME` or the equivalent of `mkdir` on your operating system.

6. Now that you've made your first change, it's helpful to know how to commit changes to the remote repository. First, check what has been changed on your local repository since your last commit by using:
    ```console
    git status
    ```
    
7. Any changes that haven't been added to the next commit will appear in red, usually with a note on what command needs to be run to add these changes.
    * For files that have been added, use:
        ```console
        git add FILE_NAME
        ```
    * For directories (i.e. folders) that have been added, use:
        ```console
        git add DIRECTORY_NAME/
        ```

        Note that your directory will only be tracked if it isn't empty. Add a README.md file with a basic description (e.g. "My code") so that you can the directory. Note that you should only add the directory with the README.md file initially, before you've added any other content. This is explained in step 8. Add the directory first, commit it, and then add any files you want to include in that directory and commit those as well.

        Note that this will also add any files that are contained in your directory as well. It is generally bad practice to add directories this way, and the reason why will be explained later in step 8. You should add your directory upon its creation when it is empty, and add any new files separately after they have been added to the directory.

    * For files that have been removed, use:
        ```console
        git rm FILE_NAME
        ```
    * For directories that have been removed, use:
        ```console
        git rm -r DIRECTORY_NAME
        ```
        Note that this will also remove all files within the directory as well. While it is bad form to add all the files in a directory at once, this doesn't hold true for removing directories that contain files.
        
8. Now commit changes you've made using:
    ```console
    git commit -m "Your commit message."
    ```
    Your commit message should briefly explain the changes that were made. Because every file modified or added should have its own commit message, you should add each file and commit it separately. This allows tracking of changes easier for those keeping tabs on the repository. For longer commit messages where the explanation should be of paragraph length, utilize the following:
    ```console
    git commit
    ```
    This will open the text editor your git is set to and creates the commit message in a file. The first line of your file is the commit header which appears beside the files on the repote repository. Add a blank line and add your in-depth description in a paragraph underneath the blank line. If you don't wish to use git's default editor (VIM through the git bash interface), you can set your editor of choice using 
    ```console
    git config --global core.editor PATH/TO/text_editor
    ```
    
9. After the changes have all been committed, push them to the remote repository. For your first push after creating a new branch, use:
    ```console
    git push -u origin NAME_OF_YOUR_BRANCH
    ```
    Every subsequent push can be made with the simplified command:
    ```console
    git push
    ```
10. If a change has been made to the remote repository that you want included on your local repository, use:
    ```console
    git pull
    ```
    
11. Lastly, to merge your branch with the master branch, create a pull request. Title your pull request and add a description that explains what has been modified and what you would like people to look over. Add someone as your reviewer and make the pull request. The reviewer will look over your code and makes comments inquiring about your code or making helpful suggestions. If the reviewer requires a change be made, make the required change on your local repository and push the changes. The pull request will always switch to the most up-to-date commit of your branch. When the reviewer is satisfied, they will approve the change. You may have compatability issues between your branch and the master branch. If that is the case, GitHub has the ability to open the files with conflicts and you can make your changes there. Alternatively, your local repository files will be edited to show where the conflicts are, and using an editor, particularly one with git compatability (e.g. PyCharm, VSCode, VS), you can resolve the conflicts on your local repo and push the changes; this is the preferred method. Once all conflicts are resolved, you can merge your branch with the master branch.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```
MIT License

Copyright (c) 2023 UW-Physics-Undergrads-of-2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
