# WORLDBANK DASHBOARD
Data visulization dashboard built using Flask, Bootstrap and Plotly. This project is part of Udacity's data science course introduction to web development.

Following the instructions below will launch data vis dashboard showing charts using world bank data.

View the dashboard using this link [here:](https://shiva-app.herokuapp.com/)

### Requirements
In terminal do to directory and use the package manager pip to install requirements.

```
cd Web Deployment
pip install -r requirements.txt
```


### Heroku Intallation as given by Udacity
2. Next, go to www.heroku.com and create an account if you haven't already.

3. Then, follow the process given in the previous video:
- update python using the terminal command `conda update python`

- create a virtual environment. Note that you can create the virtual environment inside the 5_deploy folder. But then you would end up uploading that folder to Heroku unecessarily. Consider creating the virtual environment in the workspace folder. Or alternatively, you can create a .gitignore file inside the 5_deploy folder so that the virtual enviornment folder gets ignored

- pip install the libraries needed for the web app. In this case those are flask, pandas, plotly, and gunicorn.

- next install the heroku command line tools with the following command:
curl https://cli-assets.heroku.com/install-ubuntu.sh | sh
https://devcenter.heroku.com/articles/heroku-cli#standalone-installation

- then check the installation with the command:
heroku —-version

- next, log into heroku using the command:
heroku login
and then enter your email and password when asked

- remove app.run() from the worldbank.py file

- go into the 5_deployment folder with:
cd 5_deployment

- create a procfile with the command
touch Procfile
and put the following in the Procfile
web gunicorn worldbank:app

- Then create a requirements file with this command:
pip freeze > requirements.txt

- Next, initialize a git repository with the following commands:
git init
git add .

- configure the email and user name, you can use these commands:
git config --global user.email email@example.com
git config --global user.name "my name"

- make a commit with this command:
git commit -m "first commit"

- create a uniquely named heroku app. Use this command:
heroku create my-app-name
If you get a message that the app name is already taken, try again with a different app name until you find one that is not taken

- check that heroku added a remote repository with this command:
git remote -v

- push the app to Heroku:
git push heroku master

Go to the link for your web app to see if it's working. The link should be https://app-name.heroku.com

### Acknowledgement

Udacity Instructors and Heroku










