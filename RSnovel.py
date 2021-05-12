import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import sklearn
from sklearn.decomposition import TruncatedSVD
import sys
import requests
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout
import sys
from PyQt5.QtGui import QIcon, QFont, QPixmap, QImage
from PyQt5.QtCore import QRect, QSize

book = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
book.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
user = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
user.columns = ['userID', 'Location', 'Age']
rating = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
rating.columns = ['userID', 'ISBN', 'bookRating']

combine_book_rating = pd.merge(rating, book, on='ISBN')

combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])

book_ratingCount = (combine_book_rating.
     groupby(by = ['bookTitle'])['bookRating'].
     count().
     reset_index().
     rename(columns = {'bookRating': 'totalRatingCount'})
     [['bookTitle', 'totalRatingCount']]
    )
rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount,
                                                         left_on = 'bookTitle',
                                                         right_on = 'bookTitle',
                                                         how = 'left')
popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

combined = rating_popular_book.merge(user,
                                     left_on = 'userID',
                                     right_on = 'userID',
                                     how = 'left')

us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)
us_canada_user_rating.head()

if not us_canada_user_rating[us_canada_user_rating.duplicated(['userID', 'bookTitle'])].empty:
    initial_rows = us_canada_user_rating.shape[0]

    print('Initial dataframe shape {0}'.format(us_canada_user_rating.shape))
    us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
    current_rows = us_canada_user_rating.shape[0]
    print('New dataframe shape {0}'.format(us_canada_user_rating.shape))
    print('Removed {0} rows'.format(initial_rows - current_rows))


us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle',
                                                          columns = 'userID',
                                                          values = 'bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(us_canada_user_rating_matrix)

query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1),
                                          n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i,
                                                       us_canada_user_rating_pivot.index[indices.flatten()[i]],
                                                       distances.flatten()[i]))

us_canada_user_rating.set_index('bookTitle', inplace=True)

url_book1 = us_canada_user_rating['imageUrlM'].loc[us_canada_user_rating_pivot.index[indices.flatten()[1]]][0]
url_book2 = us_canada_user_rating['imageUrlM'].loc[us_canada_user_rating_pivot.index[indices.flatten()[2]]][0]
url_book3 = us_canada_user_rating['imageUrlM'].loc[us_canada_user_rating_pivot.index[indices.flatten()[3]]][0]
url_book4 = us_canada_user_rating['imageUrlM'].loc[us_canada_user_rating_pivot.index[indices.flatten()[4]]][0]
url_book5 = us_canada_user_rating['imageUrlM'].loc[us_canada_user_rating_pivot.index[indices.flatten()[5]]][0]

class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.create_ui()

    def create_ui(self):
        self.setWindowTitle("PyQT5 window")
        self.setGeometry(100, 100, 800, 500)
        self.setWindowIcon(QIcon('pycon.png'))
        self.grid_layout()

        self.show()

    def grid_layout(self):
        grpbox = QGroupBox("What are recommended books for you, if you were looking at book '{0}'".format(us_canada_user_rating_pivot.index[query_index]))
        grpbox.setFont(QFont('Arial', 12))

        grid = QGridLayout()
        vbox = QVBoxLayout()

        image1 = QImage()
        image1.loadFromData(requests.get(url_book1).content)
        pixmap1 = QPixmap(image1)
        resized_pix1 = pixmap1.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
        img1 = QLabel()
        img1.setPixmap(resized_pix1)
        grid.addWidget(img1, 0, 0)

        image2 = QImage()
        image2.loadFromData(requests.get(url_book2).content)
        pixmap2 = QPixmap(image2)
        resized_pix2 = pixmap2.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
        img2 = QLabel()
        img2.setPixmap(resized_pix2)
        grid.addWidget(img2, 0, 1)

        image3 = QImage()
        image3.loadFromData(requests.get(url_book3).content)
        pixmap3 = QPixmap(image3)
        resized_pix3 = pixmap3.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
        img3 = QLabel()
        img3.setPixmap(resized_pix3)
        grid.addWidget(img3, 0, 2)

        image4 = QImage()
        image4.loadFromData(requests.get(url_book4).content)
        pixmap4 = QPixmap(image4)
        resized_pix4 = pixmap4.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
        img4 = QLabel()
        img4.setPixmap(resized_pix4)
        grid.addWidget(img4, 0, 3)

        image5 = QImage()
        image5.loadFromData(requests.get(url_book5).content)
        pixmap5 = QPixmap(image5)
        resized_pix5 = pixmap5.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
        img5 = QLabel()
        img5.setPixmap(resized_pix5)
        grid.addWidget(img5, 0, 4)

        text1 = QLabel()
        text1.setWordWrap(True)
        text1.setText(us_canada_user_rating_pivot.index[indices.flatten()[1]])
        grid.addWidget(text1, 1, 0)

        text2 = QLabel()
        text2.setWordWrap(True)
        text2.setText(us_canada_user_rating_pivot.index[indices.flatten()[2]])
        grid.addWidget(text2, 1, 1)

        text3 = QLabel()
        text3.setWordWrap(True)
        text3.setText(us_canada_user_rating_pivot.index[indices.flatten()[3]])
        grid.addWidget(text3, 1, 2)

        text4 = QLabel()
        text4.setWordWrap(True)
        text4.setText(us_canada_user_rating_pivot.index[indices.flatten()[4]])
        grid.addWidget(text4, 1, 3)

        text5 = QLabel()
        text5.setWordWrap(True)
        text5.setText(us_canada_user_rating_pivot.index[indices.flatten()[5]])
        grid.addWidget(text5, 1, 4)

        grpbox.setLayout(grid)
        vbox.addWidget(grpbox)

        self.setLayout(vbox)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())