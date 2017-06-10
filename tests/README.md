
Simple test, just run:
    
    py.test
    
See coverage:
    
    coverage run --source npdl -m py.test
    coverage report -m 

Or, write the coverage report into html files:
    
    coverage run --source npdl -m py.test
    coverage html -d htmlcov 
    




