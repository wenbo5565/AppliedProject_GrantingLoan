## Applied Project - Loan Granting
by Wenbo Ma

### Introduction

#### Overview

Loan interest charges is an important revenue stream for commerial banks and other financial institutions. Loan application prediction is a key component in the loan management process. In this project, we develop a machine learnning model to predict whether a loan application will be fully paid back in the future based on a bank's historical data. We show that our model's performance is better than the bank's current strategy. Additionally, based on our test results, we propose a new strategy on previously denied applications to further increase profitability. 

#### Data

The data is from a private bank. It contains all applications submitted to the bank. The label information includes whether the bank granted it and finally, whether the borrowers managed to repay it. The features include both information regarding the loan and the applicant such as loan purpose, age, annual salary and etc.

#### Rule

To compare profitability of our model to that of the bank's current strategy, following rule is assumed:

  * If our model grant a loan and it doesn't get repaid, we lose 1
  * If our model grant a loan and it gets repaid, we gain 1
  * If our model dosen't grant the loan, we gain 0

### Model Building Procedure and Analysis

Please see the project report and code link for details.

[Project Report](https://github.com/wenbo5565/AppliedProject_GrantingLoan/blob/master/Project_Report__Loan_Granting.pdf) --- Data Description, Exploratory Analysis and Model Building Procedure

[Project Code](https://github.com/wenbo5565/AppliedProject_GrantingLoan/blob/master/GrantingLoan%20core.py) --- All Python Code for the Project

### Result and Proposed Strategy

#### Profitability

Based on the scoring assumption from the **Rule** section above, our model outperforms the bank's current strategy (calculated based on the data). The performance is shown below.

| Model | Total Score | Mean Score per Loan
| --- | --- | ---- |
| Bank's Current Model | 1376 | 0.289 |
| Our Model | 2699 | 0.566 |

#### Strategy on Previously Denied Applications

<!--<img src="https://github.com/wenbo5565/AppliedProject_GrantingLoan/blob/master/attachment/fp.JPG"  height="60%" width="60%">-->

| Model Prediction| True Label| Number of Cases
| --- | --- | ---- |
| Repay | Repaid| 2859 |
| Repay | Defaulted | 160 |

From the table above, we can see that our model has much more true positive cases than false positive cases. If we grant a loan for predicted positive case, we would gain 0.89 score on average. Therefore if the bank would run our model on the previously denied applications and grants loans to those predicted repay, the bank would make more profits/scores.

#### Strategy on Data Collection

The bank's current data only contains information about an applicant's historical and current situation. The problem with the data is that we have no way to estimate the trajectory of a person. Two people could be in the same economic situation today, but their future economic trajectory can be very different. If there were more personal information, we could start predicting in a much more predictive way. For example, A CS student at a top univeristy will likely have a non great economic situation today, but his/her likelihood of being able to repay the loan in future is much higher than other people with the same current economic situation. Therefore we would recommend the bank collect more personal information.

### Conclusion

In this project, we build a gradient boosting trees model to predict if a loan application will be repaid or not. We conclude the project as follows:

 * Our proposed model outperforms the bank's current strategy given the pre-defined rule.
 * The bank may want to grant loans to those previoulsy denied applicantions to be more profitable.


