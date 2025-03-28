# Federated-Machine-Learning
Note:
Flower_initial_implemenatation is first try at utilizing flower and currently yields a variety of results
FLWR_FML is a second try at flower from scratch and trys to create a bare bones flower implementation - currently not working
FML_MVP is a minimum viable product of federated machine learning that just employs the basics using sklearn and XGBoost - no other framework but the implementation does work

To run:
in ..\Federated Machine Learning GitHub\Federated-Machine-Learning\Federated Machine Learning XGboost
run the command pip install -e
this will install all dependencies

To run the Fed Learning in the same directory as the install:
use the command flwr run .

This will run the simulation and the server will log to logfile, the clients will log to logfile_client_0 and logfile_client_1 since the simulation is set up for two clients on the fridge data