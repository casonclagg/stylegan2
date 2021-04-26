convert dad.png average.png -size 1024x50 xc:White +swap -background White -gravity South +append temp.png && convert temp.png mom.png -size 1024x50 xc:White +swap -background White -gravity South +append parents.png

montage -mode concatenate -tile 5x "kid*.png" kids.jpg

convert parents.png kids.png -gravity center -append final.jpg