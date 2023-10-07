import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open('shopping.csv') as csv_file:
        evidence = []
        labels = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                row[10] = mtoi(row[10])
                row[15] = visitirtypetoint(row[15])
                row[16] = FTtoInt(row[16])
                row[17] = FTtoInt(row[17])
                tofloat = [ 1, 3, 5, 6, 7, 8, 9]
                toint = [0, 2, 4, 11, 12, 13, 14]
                for i in toint:
                    row[i] = int(row[i])
                for i in tofloat:
                    row[i] = float(row[i])
                 
                evidence.append(row[ 0 : 17 ]) 
                labels.append(row[17])
                line_count += 1
    
    return(evidence, labels)


def FTtoInt(s):
    if s == "FALSE":
        return(0)
    else:
        return(1)
def visitirtypetoint(s):
    if s == 'Returning_Visitor':
        return (1)
    else:
        return(0)

def mtoi (x):
    months = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr':4,
         'may':5,
         'jun':6,
         'jul':7,
         'aug':8,
         'sep':9,
         'oct':10,
         'nov':11,
         'dec':12
        }
    a = x.strip()[:3].lower()
    try:
        ez = months[a]
        return (ez)
    except:
        raise ValueError('Not a month')





def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(evidence, labels)
    return(knn)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    negetives = 0
    positives = 0
    negetives_s = 0
    positives_s = 0
    for i in range (0, len(labels)):
        if labels[i] == 1:
            positives +=1
            if predictions[i] == 1:
                positives_s+=1
        else:
            negetives +=1
            if predictions[i] == 0:
                negetives_s +=1
    
    sensitivity = positives_s / float(positives)
    specificity = negetives_s / float(negetives)

    return(sensitivity, specificity)
    



if __name__ == "__main__":
    main()
