# curls
## Completely Unsupervised Reinforcement Learning Server

Control reinforcement learning training and visualization via REST API.  This is (clearly) a work in progress.

You can, however, install it using pip:
`pip install curls`

To build from this repo, the recommended approach would be to build a Python 3.6 environment using Anaconda:
`conda create -n curls python=3.6`

and installing the requirements via pip:
`pip install -r requirements.txt`

The SessionManager uses SQLAlchemy to store data.  It's been tested with Postgres, and requires JSON data types, but otherwise shouldn't be dependent on the database you use.  With that being said, you can install Postgres and create a database via:

`sudo apt install postgresql postgresql-contrib -y
sudo -u postgres createdb <db_name>`

The SessionManager will look for a `DATABASE_URL` variable in `curls/framework/.env`.  Have this point to the database you want to use, e.g.,

`DATABASE_URL='postgresql://postgres:postgres@localhost/<db_name>'`

You can run the development API by navigating to `curls/framework` and running `python app.py`.  This runs a Flask API which can be accessed at `localhost:5000`.

You can run the development client by navigating to `curls/client` and running `npm start`.  This runs a React app which can be accessed at `localhost:3000`.  It expects the API to be running at `localhost:3000`.

To create and start a training session, you can run the `SessionManager.py` file directly, e.g., 

`python SessionManager.py --create-all -e CartPole-v1 -ep 25 -s 100`

Here, `--create-all` will initialize the database tables, and only needs to be run on a fresh database.  `-e CartPole-v1` is the Gym Environment we want to use, `-ep 25` says to run 25 episodes per session, and `-s 100` says to run it for 100 sessions.  You can run `python SessionManager.py -h` to list all options.

A rough diagram of the general idea can be found <a href="/curls/framework/architecture.pdf"> here</a> (note: this diagram is most likely not accurate).