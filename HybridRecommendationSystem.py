
#Ana amaç =>>>  ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini kullanarak 10 film önerisi yapınız.

#1) User Based Recommendation System
#Görev1
#Adım1

import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv('W5/recommender_systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('W5/recommender_systems/datasets/movie_lens_dataset/rating.csv')

#Adım2
df = movie.merge(rating, how="left", on="movieId")
df.head()

#Adım3
comment_counts=pd.DataFrame(df["title"].value_counts())
common_comments=df[~df["title"].isin(comment_counts[comment_counts.title<=1000].index)]

#Adım4
user_movie_df=common_comments.pivot_table(index="userId",columns="title",values="rating")

#Adım5
#Tüm işleri fonksiyonlaştır
def create_user_movie_df():
    movie = pd.read_csv('W5/recommender_systems/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('W5/recommender_systems/datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    common_comments = df[~df["title"].isin(comment_counts[comment_counts.title <= 1000].index)]
    user_movie_df = common_comments.pivot_table(index="userId", columns="title", values="rating")
    return user_movie_df

user_movie_df=create_user_movie_df()


#**************************Görev2******************************
#Adım1
#Rasgele bir kullanıcı seçilir.

Random_user=int(pd.DataFrame(user_movie_df.index).sample(1,random_state=45).values)

#Adım2

random_user_df = user_movie_df[user_movie_df.index == Random_user]

#Adım3
movies_watched=random_user_df.columns[random_user_df.notna().any()].to_list()

#****************************Görev3***********************************

#Adım1

movies_watched_df=user_movie_df[movies_watched]

#Adım2

user_movie_count=pd.DataFrame(movies_watched_df.T.notnull().sum().reset_index())
user_movie_count.columns = ["userId", "movie_count"]
#Adım3

len(movies_watched)
user_movie_count["Ratio"]=user_movie_count["movie_count"]*100/len(movies_watched)
user_movie_count["Ratio"]=user_movie_count["Ratio"].round(0)

#Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı id’lerinden users_same_movies adında bir liste oluşturulur
users_same_movies=user_movie_count[user_movie_count["Ratio"]>=60]["userId"]

#Görev4

#Adım1

final_df =pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],random_user_df[movies_watched]])

#Adım2

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
#Adım3

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"]== Random_user ) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

#Adım4
top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

#Adım4

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != Random_user]

#Görev5

#Adım1

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

#Adım2

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

#Adım3

recommend_movies=recommendation_df[recommendation_df["weighted_rating"]>3.5].sort_values(by='weighted_rating',ascending=False)

#Adım4
recommend_movies.merge(movie, how="inner", on="movieId").head(5)

#2) Item Based Recommendation

#Adım2

Random_user=int(pd.DataFrame(df.userId).sample(1,random_state=45).values)
new_df=df[(df['userId']==Random_user) & (df["rating"]==5)]
movie_id=int(new_df.sort_values(by="timestamp",ascending=False)["movieId"].head(1).values)

#Adım3
comment_counts = pd.DataFrame(df["title"].value_counts())
common_comments = df[~df["title"].isin(comment_counts[comment_counts.title <= 1000].index)]
user_movie_df = common_comments.pivot_table(index="userId", columns="title", values="rating")

#Adım4
movie_name=movie[movie["movieId"]==movie_id]["title"].values[0]
movie_name_df = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name_df).sort_values(ascending=False).head(10)


