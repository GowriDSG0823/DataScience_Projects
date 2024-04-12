create database library_managemet_system;
use library_managemet_system;

# Create table called "books"
create table books (
book_id int Primary Key auto_increment,
title varchar (100),
author varchar (100),
genre varchar (100),
publication_date date ,
isbn varchar(13),
quantity_available int);


insert into books(book_id,title,author,genre,publication_date,isbn,quantity_available)
values (1,'Harry Potter','J.K. Rowling','Fantasy','1997-06-26','9788700631625',5),
(2,'To Kill a Mockingbird','Harper Lee','Fiction','1960-07-11','9780061120084',3),
(3,'The Great Gatsby','F. Scott Fitzgerald','Classic','1925-04-10','9780743273565',2),
(4,'The Catcher in the Rye', 'J.D. Salinger', 'Fiction', '1951-07-16', '9780316769488', 7),
(5,'Pride and Prejudice', 'Jane Austen', 'Romance', '1813-01-28', '9780141439518', 4),
(6,'1984', 'George Orwell', 'Dystopian', '1949-06-08', '9780451524935', 6),
(7,'The Hobbit', 'J.R.R. Tolkien', 'Fantasy', '1937-09-21', '9780261102217', 8),
(8,'Moby-Dick', 'Herman Melville', 'Adventure', '1851-10-18', '9780142437247', 5),
(9,'To the Lighthouse', 'Virginia Woolf', 'Modernist', '1927-05-05', '9780156030472', 3),
(10,'Frankenstein', 'Mary Shelley', 'Gothic', '1818-01-01', '9780486282114', 6),
(11,'Animal Farm', 'George Orwell', 'Satire', '1945-08-17', '9780451526342', 6)
;
select* from books;

# Create table called "authors"
create table authors (
author_id int primary key auto_increment,
author_name varchar(100));
insert into authors (author_name)
values ('J.K. Rowling'),('Harper Lee'),('F. Scott Fitzgerald'),
('J.D. Salinger'),('Jane Austen'),('George Orwell'),
('J.R.R. Tolkien'),('Herman Melville'),('Virginia Woolf'),('Mary Shelley')
;
select * from authors;

# Create table called "members"
create table members (
member_id int primary key auto_increment,
member_name varchar(100),
member_email varchar(100), 
member_phone varchar(12));
INSERT INTO members (member_name, member_email, member_phone)
VALUES
    ('John Doe', 'john.doe@example.com', '123-456-7890'),
    ('Jane Smith', 'jane.smith@example.com', '987-654-3210'),
    ('Michael Johnson', 'michael.johnson@example.com', '456-789-0123'),
    ('Emily Brown', 'emily.brown@example.com', '789-012-3456'),
    ('Christopher Davis', 'christopher.davis@example.com', '234-567-8901'),
    ('Sarah Wilson', 'sarah.wilson@example.com', '890-123-4567'),
    ('Matthew Taylor', 'matthew.taylor@example.com', '345-678-9012'),
    ('Amanda Martinez', 'amanda.martinez@example.com', '901-234-5678')
    ;
select * from members;

# Create table called "borrowings"
CREATE TABLE borrowings (
    borrowing_id INT PRIMARY KEY AUTO_INCREMENT,
    book_id INT,
    member_id INT,
    borrowing_date DATE,
    return_date DATE,
    is_returned BOOLEAN,
    FOREIGN KEY (book_id) REFERENCES books(book_id),
    FOREIGN KEY (member_id) REFERENCES members(member_id)
);
insert into borrowings values 
(1,1,1,'2024-02-10','2024-02-20',true),
(2,2,2,'2024-02-15',NULL,false),
(3, 3,1, '2024-02-18', '2024-03-02', TRUE),
(4, 4,8, '2024-02-20', '2024-03-05', TRUE),
(5, 1,6, '2024-02-22', null , FALSE),
(6, 1,8, '2024-02-25', '2024-03-15', FALSE),
(7, 11,7, '2024-02-28', '2024-03-20', TRUE),
(8, 5,4, '2024-03-01', null, FALSE),
(9, 1,2, '2024-03-03', '2024-03-30', TRUE),
(10, 8,1, '2024-03-05', '2024-04-01', FALSE),
(11, 11,6, '2024-03-08', '2024-04-05', TRUE),
(12, 7,6, '2024-03-10', '2024-04-10', TRUE),
(13, 4, 8,'2024-03-12', '2024-04-15', FALSE),
(14, 11,3, '2024-03-15', null, TRUE),
(15, 9,4, '2024-03-18', '2024-04-25', FALSE),
(16, 10, 1,'2024-03-20', '2024-04-30', TRUE)
 ;   
 select * from borrowings;

 # Create table called "Publishers Table"
 create table publishers (publisher_id int primary key,
 publisher_name varchar(100), publisher_country varchar (100),
 book_id int, foreign key (book_id) references books(book_id),
 author_id int, foreign key (author_id) references authors(author_id)
 );
 insert into publishers 
 (publisher_id,publisher_name,publisher_country,book_id,author_id)
 values 
 (1,'Penguin Random House','United States',1,1),
 (2	,'HarperCollins','United Kingdom',2,2),
 (3,'Simon & Schuster', 'United States',3,3),
(4,'Macmillan Publishers', 'United States',6,6),
(5,'Hachette Livre', 'France',5,5),
(6,'Pearson Education', 'United Kingdom',11,1),
(7,'Scholastic Corporation', 'United States',8,8)
;
select * from publishers;

# Create table called "book_copies"
CREATE TABLE book_copies (
    copy_id INT PRIMARY KEY,
    book_id INT,
    FOREIGN KEY (book_id) REFERENCES books(book_id),
    copy_number INT,
    book_condition VARCHAR(10),
    shelf_location VARCHAR(5)
);


insert into book_copies values
(1,	1,001,'Good','A1'),
(2,	1,002,'Fair','B3'),
(3, 4,003,'Excellent','C2'),
(4,11,004, 'Good', 'A2'),
(5,10,005, 'Fair', 'B1'),
(6,3,006, 'Excellent', 'C3'),
(7,3,007, 'Good', 'A3'),
(8,11,008, 'Fair', 'B2'),
(9,2,009,'Excellent', 'C1');
select * from book_copies;

# Create table called "authors_books_map" 
create table authors_books_map ( author_book_id int primary key,
author_id int, foreign key (author_id) references authors(author_id),
book_id INT,
FOREIGN KEY (book_id) REFERENCES books(book_id));
insert into authors_books_map values (1,1,1),(2,2,2),(3,3,3),(4,4,4),
(5,5,5),(6,6,6),(7,7,7),(8,8,8),(9,9,9),(10,10,10),(11,1,11);
select * from authors_books_map ;

# Create table called "reviews"
create table reviews ( review_id int primary key,
book_id int, foreign key (book_id) references books(book_id),
member_id int, foreign key (member_id) references members(member_id),
rating float, review_text varchar(200), review_date date);
insert into reviews values 
	(1,1,1,4.5,"A classic masterpiece.",'2024-02-12'),
    (2,2, 3, 3.8, 'An interesting read.', '2024-02-15'),
    (3,3, 2, 4.0, 'Loved it!', '2024-02-18'),
    (4,1, 4, 4.2, 'Highly recommend.', '2024-02-20'),
    (5,4, 5, 3.5, 'Not bad, but could be better.', '2024-02-22'),
    (6,2, 6, 4.8, 'Couldn''t put it down!', '2024-02-25'),
    (7,5, 7, 3.0, 'Didn''t meet my expectations.', '2024-02-28'),
    (8,6, 8, 4.6, 'Absolutely loved this book!', '2024-03-01'),
    (9,7, 4, 3.7, 'A bit disappointing.', '2024-03-05'),
    (10,8, 1, 4.3, 'Engaging storyline.', '2024-03-08');
select * from reviews;

# Create table called "transactions"
create table transactions ( transaction_id int primary key,
member_id int, foreign key (member_id) references members(member_id),
transactions_date date, transaction_type varchar (100), 
amount_paid varchar (50)
);
insert into transactions values
(1,1,'2024-02-10',	'borrow',	0),
(2,1,'2024-02-20',	'return',	0),
(3,8,'2024-02-15','purchase',5),
(4, 2, '2024-02-25', 'borrow',0),
(5, 3, '2024-03-01', 'purchase', 10),
(6, 8, '2024-02-15', 'purchase', 5);
select * from transactions;



-- Queries --
-- 1.	List all books borrowed by a specific member:
select books.book_id, books.title, members.member_id, members.member_name 
from borrowings 
join books on borrowings.book_id=books.book_id
join members on borrowings.member_id=members.member_id 
 where members.member_id=1;

-- 2.	Find the most popular genres:
select books.genre, count(borrowings.book_id) max_borrowed from borrowings
join books on borrowings.book_id=books.book_id
group by borrowings.book_id order by borrowings.book_id desc
limit 1;

-- 3.	Identify books with the highest average rating:
select books.title, avg(reviews.rating) average_rating from reviews
join books on reviews.book_id = books.book_id
group by books.title order by average_rating desc limit 1;

-- 4.	List all members who have borrowed more than 3 books:
select members.member_name, count(book_id) as max_books_borrowed 
from borrowings 
join members on borrowings.member_id=members.member_id 
group by members.member_name having max_books_borrowed >3;

-- 5.	List all members who have borrowed less than 5 books:
select members.member_name, count(book_id) as max_books_borrowed 
from borrowings 
join members on borrowings.member_id=members.member_id 
group by members.member_name having max_books_borrowed <5;

-- 6.	Retrieve the top-rated books with at least 5 reviews:
select books.title, max(reviews.rating) max_rating,
 count(reviews.book_id) no_of_reviews
from reviews join books on reviews.book_id=books.book_id  
group by books.title 
having count(reviews.book_id) >= 5
order by books.title desc, max_rating,no_of_reviews desc limit 3;

-- 7.	Calculate the total revenue generated from book purchases:
select sum(amount_paid) total_revenue from transactions;

-- 8.	List all books with their respective authors and publishers:
select books.title,authors.author_name, publishers.publisher_name
from publishers 
join books on books.book_id=publishers.book_id
join authors on authors.author_id = publishers.author_id;

-- 9.	Find books that are currently available for borrowing:
select distinct books.title from books join borrowings on
borrowings.book_id=books.book_id 
where borrowings.return_date is not null;

-- 10.	Identify members who have overdue books:
select members.member_name from members join borrowings on
borrowings.member_id=members.member_id 
where borrowings.return_date is null;

-- 11.	List the top 10 most borrowed books:
select count(borrowings.book_id) num_borrowed , books.title from borrowings
join books on borrowings.book_id=books.book_id
group by books.title order by num_borrowed desc limit 10;

-- 12.	Calculate the average number of days a book is borrowed for:
select books.title, 
avg(datediff(borrowings.return_date,borrowings.borrowing_date)) as no_of_days_borrowed
from borrowings
join books on borrowings.book_id=books.book_id 
group by books.title;

-- 13.	Find the total number of books published in each year:
select year(publication_date) years, count(book_id) from books
group by years order by years desc;

-- 14.	Identify members who have borrowed books more than once:
select members.member_name, count(borrowings.member_id)
from borrowings join members on 
members.member_id=borrowings.member_id 
group by members.member_name
having count(borrowings.member_id)>1
order by count(borrowings.member_id) desc ;

-- 15.	List all books with their respective authors and average ratings:
select books.title, books.author, avg(reviews.rating) avg_rating
from books join reviews on books.book_id=reviews.book_id
group by books.title, books.author
order by avg(reviews.rating) desc;

-- 16.	Calculate the total number of copies available for each book:
select * from book_copies;
select books.title, count(book_copies.book_id) total_copies
from books join book_copies on
books.book_id=book_copies.book_id
group by books.title order by count(book_copies.book_id) desc;

-- 17.	Create a view of transaction table and provide privilege to another user. The user can view only member id and transaction date and privilege should be to select id who made transaction on any specific date.
 create view transactions_1 as 
 select member_id, transactions_date
 from transactions;

 select * from transactions_1;
 
 create user 'Amy'@'localhost' identified by 'Amy@012';
 grant select on transactions_1 to 'Amy'@'localhost';
 











