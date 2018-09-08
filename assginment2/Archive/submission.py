import sys
import math
import matplotlib.pyplot as plt
import pandas as pd


# check if the user_id user input valid or not
def user_id_check(user_predicted, userlist):
    if user_predicted not in userlist.keys():
        sys.exit("This user id does not exist in database!\nYou could try 1, 214...")


# check if the book isbn user input valid or not
def movie_id_check(movie, userlist):
    for i in userlist.items():
        if movie in i[1]:
            break
    else:
        sys.exit("This book does not exists in database!\nYou could try 20, 141...")


# process data, make an dictionary to store all the book ratings for each user.
# dictionary looks like {user:{book: rate,...},...}
def process_trainning_data(dataset):
    userlist = {}
    for i in dataset.itertuples():
        user = str(i[1])
        movie = str(i[2])
        rate = str(i[3])
        if user not in userlist:
            rating_items = {}
            rating_items[movie] = rate
            userlist[user] = rating_items
        else:
            rating_items = userlist.get(user)
            rating_items[movie] = rate
            userlist[user] = rating_items
    return userlist


# calculate average rating for each user
def average_of_user(userlist):
    average = {}
    for i in userlist.items():
        num, total_rate = 0, 0
        for temp in i[1].items():
            total_rate += float(temp[1])
            num+=1
        average[i[0]] = float(total_rate/num)
    return average


# use Pearson correlation coefficient to represent the relation between two different items
# p = cov(x, y)/(var(x) * var(y))^(1/2), cov is the covariance and var is standard deviation
def nearest_neighbours(user_predicted, userlist):
    average = average_of_user(userlist)
    relation = {}
    for i in userlist.items():
        # calculate pearson correlation coefficient
        # cov(X,Y)=E((X-average(X))*(Y-average(Y)))
        if i[0] != user_predicted:
            cov = 0.0
            var_x, var_y = 0.0, 0.0
            for rate_from_user_id in userlist.get(user_predicted).items():#books which user rating
                if rate_from_user_id[0] in userlist.get(i[0]).keys():# another one called i also rate this book
                    x_ave = float(rate_from_user_id[1])-average.get(user_predicted)
                    y_ave = float(userlist.get(i[0]).get(rate_from_user_id[0])) - average.get(i[0])
                    cov += x_ave * y_ave
                    var_x += x_ave**2
                    var_y += y_ave**2
            if var_x != 0 and var_y != 0:
                relation[i[0]] = float(cov/(math.sqrt(var_x) * math.sqrt(var_y)))
    neighbours = {k:relation[k] for k in sorted(relation, key=relation.get, reverse=True) if relation[k] != 0}
    return neighbours, relation


# get top-k nearest neighbours
def k_nearest_neighbour(neighbours, relation, knn):
    neighbours_ = [k for k in sorted(relation, key=relation.get, reverse=True) if relation[k] != 0][:int(knn)]
    return {p: neighbours[p] for p in neighbours_}


# get weight of each common movie, apply the weight to get final prediction score
def predict(n_nearest_neighbour, movie, userlist):
    Prediction = 0
    weight = {}
    for i in n_nearest_neighbour.items():
        if movie in userlist.get(i[0]):
            weight[i[0]] = i[1]

    x, y = 0, 0
    for i in weight.items():
        x += float(userlist.get(i[0]).get(movie)) * float(i[1])
        y += float(i[1])
    if y != 0:
        Prediction = x/y
    return int(round(Prediction, 0))


# using n test samples to test this model by applying top-knn nearest neighbour.
# return the accuracy of this model
def accuracy(userlist, test, n, knn):
    count_accurate = 0
    invalid_test = 0
    count = 0
    for each_test in test.itertuples():
        user_test = str(each_test[1])
        item_test = str(each_test[2])
        rate_test = str(each_test[3])
        if user_test not in userlist.keys():
            invalid_test += 1
        else:
            count += 1
            ## all neighbours apply
            if knn == -1:
                neighbours = nearest_neighbours(user_test, userlist)[0]
            else:
                all_neighbours, relation = nearest_neighbours(user_test, userlist)
                neighbours = k_nearest_neighbour(all_neighbours, relation, knn)
            rating = predict(neighbours, item_test, userlist)
            if int(rating) == int(rate_test):
                count_accurate += 1
            if count == n:
                accuracy = count_accurate / count
                return accuracy
        # print(invalid_test, count, count_accurate, rating, rate_test, count_accurate/count)
    # accuracy = count_accurate/(len(test) - invalid_test)
    # return round(accuracy * 100, 2)


df = pd.read_csv("ml-100k/u.data", sep='\t')
total_userlist = process_trainning_data(df)
predict_user = input("Please input a user id you want to predict: ") # e.g. 1, 214...
user_id_check(predict_user, total_userlist)
predicted_movie = input("Please input a movie id you want to predict: ") # e.g. 20, 141...
movie_id_check(predicted_movie, total_userlist)

nearest_neighbours_of_prodicted = nearest_neighbours(predict_user, total_userlist)[0]
rating_predict = predict(nearest_neighbours_of_prodicted, predicted_movie, total_userlist)
print("The user {} and movie {} contribute to {} rating".format(predict_user, predicted_movie, rating_predict))

print("Now, we analyse the influence of changes of knn on the accuracy of this collaborative-based recommendation system... ")
# choose random 80% of data to train
train = df.sample(frac=0.8, random_state=100)
test = df.drop(train.index)
userlist = process_trainning_data(train)
accuracy_with_all_neighbours = accuracy(userlist, test, 200, -1)
print("Accuracy with all neighbours is {:.2%}\n".format(accuracy_with_all_neighbours))

# k values candidates: 100, 200, 300, 400
x = [100, 200, 300, 400]
y = []
for knn in x:
    accuracy_ = accuracy(userlist, test, 200, knn)
    print("Accuracy with {} neighbours is {:.2%}\n".format(knn, accuracy_))
    y.append(accuracy_)
plt.title("Accuracy in 200 test samples with different top-k nearest neighbours")
plt.axhline(y=accuracy_with_all_neighbours, color='red')
plt.plot(x, y)
plt.legend(['all neighbours', 'top-x neighbours'], loc='upper left')
plt.show()
