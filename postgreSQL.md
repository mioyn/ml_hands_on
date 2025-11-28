

### Logical Flow of SQL Query Execution in PostgreSQL
| Step   | Clause           | What happens                              |
| ------ | ---------------- | ----------------------------------------- |
| **1**  | `FROM`           | Selects the source tables and joins them. |
| **2**  | `ON`             | Applies join conditions (if joins exist). |
| **3**  | `JOIN`           | Combines rows from multiple tables.       |
| **4**  | `WHERE`          | Filters rows *before* grouping.           |
| **5**  | `GROUP BY`       | Groups rows into buckets.                 |
| **6**  | `HAVING`         | Filters groups created by GROUP BY.       |
| **7**  | `SELECT`         | Selects columns and expressions.          |
| **8**  | `DISTINCT`       | Removes duplicates (after SELECT).        |
| **9**  | `ORDER BY`       | Sorts the results.                        |
| **10** | `LIMIT / OFFSET` | Returns only a subset of rows.            |


### Drop and create table
```sql
DROP TABLE IF EXISTS public.sales_data;

CREATE TABLE sales_data (
	sale_id INT,
	customer_id VARCHAR(10),
	product VARCHAR(50),
	region VARCHAR(10),
	amount NUMERIC(10, 2),
	sale_date DATE
);
```
### Data to execute below sql

```javascript
sale_id,customer_id,product,region,amount,sale_date
101,C001,Laptop,North,900,2024-01-01
102,C002,Phone,South,450,2024-01-02
103,C003,Tablet,East,700,2024-01-03
104,C004,Headphones,West,250,2024-01-04
105,C005,Camera,North,1200,2024-01-05
106,C001,Laptop,South,500,2024-01-06
107,C002,Phone,East,800,2024-01-07
108,C003,Tablet,West,650,2024-01-08
109,C004,Headphones,North,1100,2024-01-09
110,C005,Camera,South,400,2024-01-10
111,C001,Laptop,East,950,2024-01-11
112,C002,Phone,West,350,2024-01-12
113,C003,Tablet,North,600,2024-01-13
114,C004,Headphones,South,1000,2024-01-14
115,C005,Camera,East,700,2024-01-15
116,C001,Laptop,West,300,2024-01-16
117,C002,Phone,North,750,2024-01-17
118,C003,Tablet,South,1250,2024-01-18
119,C004,Headphones,East,400,2024-01-19
120,C005,Camera,West,850,2024-01-20
```

### 1. Show all records from the sales_data table.

```sql
select * from sales_data;
```

### 2. Retrieve the first 10 rows from the sales_data table.

```sql
select * from sales_data LIMIT 10;
```

### 3. List all unique regions where sales occurred.

```sql
select DISTINCT region from sales_data;
```

### 4. List all distinct products sold.

```sql
select DISTINCT product from sales_data;
```

### 5. Show all sales made in the North region.

```sql
select * from sales_data where region = 'North';
```

### 6. Retrieve all sales where the amount is greater than 500.

```sql
select * from sales_data where amount > 500;
```

### 7. Find all sales that occurred on January 1, 2024.

```sql
select * from sales_data where sale_date = '2024-01-01';
```

### 8. Find laptop sales made in the South region.

```sql
select * from sales_data where region = 'South' AND product = 'Laptop';
```

### 9. Show all sales made in either the East or West region.

```sql
select * from sales_data where region = 'East' OR region = 'West';
```

### 10. Retrieve all sales except those where the product is Mobile.

```sql
select * from sales_data where product != 'Mobile';
```

### 11. Display all sales sorted by amount from highest to lowest.

```sql
select * from sales_data order by amount desc;
```

### 12. Show the earliest 10 sales based on sale date.

```sql
select * from sales_data order by sale_date limit 10;
```

### 13. Retrieve all sales where the product name starts with the letter ‘P’.

```sql
select * from sales_data where product like 'P%';
```

### 14. Show all sales where the region name contains the letter ‘a’.

```sql
select * from sales_data where region like '%a%';
```

### 15. Retrieve all sales made in North or East regions.

```sql
select * from sales_data where region in ('North', 'East');
```

### 16. Find all sales where the amount is between 200 and 800.

```sql
select * from sales_data where amount between 200 and 800;
```

### 17. Retrieve sales made between January 1 and January 15, 2024.

```sql
select * from sales_data where sale_date between '2024-01-01' and '2024-01-15';
```

### 18. Calculate the total sales amount.

```sql
select sum(amount) from sales_data;
```

### 19. Find the average sale amount.

```sql
select avg(amount) from sales_data;
```

### 20. Count the total number of sales records.

```sql
select count(*) from sales_data;
```

### 21. How many distinct products were sold?

```sql
select count(DISTINCT product) from sales_data;
```

### 22. What is the highest sales amount?

```sql
select max(amount) from sales_data;
```

### 23. What is the lowest sales amount?

```sql
select min(amount) from sales_data;
```

### 24. Calculate total sales amount for each region.

```sql
select region, SUM(amount) from sales_data group by region;
```

### 25. Count how many times each product was sold.

```sql
select product, count(product) from sales_data group by product;
```

### 26. Find the average sales amount for each region.

```sql
select region, avg(amount) as avg_amt from sales_data group by region;
```

### 27. Show regions whose total sales exceed 3000.

```sql
select region, sum(amount) as sal_amt
from sales_data
group by region
having sum(amount) > 3000;
```

### 28. List products that were sold more than 2 times.

```sql
select product, count(product)
from sales_data
group by product
having count(product) > 2;
```

### 29. Find all North region sales where the product starts with ‘C’, sorted by amount descending.

```sql
select * from sales_data
where region = 'North' and product like 'C%'
order by amount desc;
```

### 30. Retrieve sales over 500 from East or West regions.

```sql
select * from sales_data
where amount > 500 and region in ('East', 'West');
```

### 31. Find all Phone sales between January 1 and January 10, 2024.

```sql
select * from sales_data
where product = 'Phone'
and sale_date between '2024-01-01' and '2024-01-10';
```

### 32. Find the top 3 customers by total purchase amount.

```sql
select customer_id, sum(amount) as tot_sum
from sales_data
group by customer_id
order by tot_sum desc
limit 3;
```

### 33. Show total and average sales amount for each region.

```sql
select sum(amount) as tot_sl, AVG(amount) as avg_sal, region
from sales_data
group by region;
```

### 34. Calculate total sales for each month.

```sql
select EXTRACT(MONTH from sale_date) as mon, sum(amount)
from sales_data
group by mon;
```

### 35. Calculate daily total sales by sale date.

```sql
select EXTRACT(day from sale_date) as dy, sum(amount)
from sales_data
group by sale_date;
```
## More...!!!!

```
customer_id,customer_name,region,signup_date
101,Anna 1,West,2021-11-30
102,Oliver 2,Central,2020-08-28
103,Emma 3,West,2021-04-16
104,Anna 4,Central,2023-03-15
105,Anna 5,North,2020-10-26
106,Mia 6,Central,2020-12-18
107,David 7,West,2021-09-03
108,Chris 8,East,2023-08-08
109,Oliver 9,South,2023-07-16
110,John 10,West,2022-12-21
111,Anna 11,West,2020-09-23
112,Laura 12,East,2022-10-02
113,Emma 13,North,2021-09-23
114,Laura 14,South,2023-06-24
115,Anna 15,Central,2021-12-20
116,Michael 16,East,2022-09-20
117,Anna 17,East,2023-04-18
118,David 18,Central,2023-02-03
119,John 19,North,2023-02-10
120,Chris 20,Central,2021-08-22
121,Anna 21,East,2023-05-24
122,John 22,West,2022-01-27
123,Oliver 23,South,2022-02-20
124,Laura 24,North,2021-07-09
125,David 25,East,2021-06-20
126,John 26,East,2023-06-30
127,David 27,South,2022-12-02
128,Chris 28,North,2022-07-31
129,Oliver 29,Central,2021-11-28
130,Chris 30,South,2021-04-14
131,Anna 31,North,2022-05-31
132,Laura 32,South,2022-01-26
133,Sarah 33,North,2021-04-21
134,David 34,North,2022-05-03
135,Anna 35,Central,2022-08-04
136,Anna 36,East,2020-08-26
137,Oliver 37,North,2021-08-13
138,David 38,East,2021-06-12
139,Chris 39,North,2021-09-27
140,John 40,South,2020-02-04
```
```
order_id,customer_id,product_id,quantity,sale_date,amount
1,101,550,2,2021-06-13,753.18
2,120,528,1,2023-08-27,183.76
3,109,508,5,2023-04-09,781.4
4,109,512,3,2022-05-25,954.45
5,133,537,2,2022-04-07,849.3
6,137,538,5,2023-05-25,1781.9
7,124,540,4,2021-08-27,1771.6
9,138,542,4,2021-11-18,1142.44
10,127,511,5,2022-01-08,2281.3
11,101,542,4,2021-03-27,1142.44
12,108,509,1,2022-10-10,259.48
13,101,505,5,2022-09-23,826.05
14,117,516,4,2021-06-25,1853.64
15,122,523,2,2023-01-17,209.24
16,139,544,2,2022-01-28,162.9
17,133,514,2,2023-03-23,558.48
18,131,506,3,2021-10-07,495.27
19,137,503,2,2021-10-24,109.84
20,117,548,5,2021-05-21,773.45
21,103,516,3,2022-11-29,1390.23
22,124,527,2,2022-01-08,647.4
23,118,504,5,2021-12-15,621.15
24,129,514,3,2021-02-02,837.72
25,128,531,4,2021-01-01,1379.4
26,111,529,3,2022-01-15,1239.6
27,103,537,5,2023-08-12,2123.25
28,134,539,2,2022-01-27,942.86
29,120,535,1,2021-06-14,294.9
30,130,529,1,2023-04-19,413.2
31,138,529,4,2023-04-23,1652.8
33,134,537,2,2022-08-11,849.3
34,138,545,3,2021-08-01,186
35,116,525,2,2023-03-18,83.86
36,121,529,1,2023-01-03,413.2
37,122,520,3,2023-05-19,137.37
39,105,516,3,2023-04-30,1390.23
40,115,544,3,2022-12-24,244.35
41,120,537,5,2023-06-18,2123.25
42,128,523,1,2023-09-24,104.62
43,123,533,5,2022-09-06,2199.15
44,102,548,1,2021-08-22,154.69
45,140,536,3,2021-08-17,1114.92
46,115,543,1,2023-02-14,471.68
48,126,502,3,2021-06-29,431.88
49,132,508,5,2023-04-06,781.4
50,120,522,1,2023-02-23,68.65
51,139,532,2,2021-02-04,552.36
52,127,549,3,2023-03-25,398.58
53,134,503,1,2022-08-20,54.92
54,136,512,2,2021-06-15,636.3
55,115,517,1,2023-04-09,158.77
56,127,502,1,2023-01-30,143.96
57,138,541,2,2022-10-03,182.42
58,103,540,1,2022-10-12,442.9
59,128,523,3,2023-06-09,313.86
60,104,509,4,2022-03-28,1037.92
61,112,536,1,2021-12-25,371.64
62,109,511,2,2022-03-01,912.52
63,110,516,1,2021-04-10,463.41
64,128,536,2,2021-02-04,743.28
65,133,517,2,2021-01-18,317.54
66,125,523,1,2021-12-16,104.62
67,118,541,5,2023-09-09,456.05
68,111,520,2,2021-04-12,91.58
69,130,539,4,2021-05-03,1885.72
70,132,548,1,2021-10-22,154.69
71,137,508,5,2021-10-30,781.4
72,106,532,2,2023-06-02,552.36
73,117,542,4,2023-03-22,1142.44
74,122,524,4,2021-07-08,290.44
75,114,503,1,2021-09-27,54.92
77,111,510,2,2023-03-30,261.76
78,137,510,2,2021-12-31,261.76
79,120,515,5,2023-04-27,1062.4
```
```
product_id,product_name,category,price
501,Headphones 1,Electronics,340.24
502,Tennis Racket 2,Sports,143.96
503,Mixer 3,Books,54.92
504,Tennis Racket 4,Home,124.23
505,Vacuum Cleaner 5,Home,165.21
506,Laptop 6,Kitchen,165.09
507,Camera 7,Home,419.05
508,Novel 8,Sports,156.28
509,Headphones 9,Books,259.48
510,Vacuum Cleaner 10,Home,130.88
511,Smartwatch 11,Electronics,456.26
512,Novel 12,Kitchen,318.15
513,Laptop 13,Sports,426.4
514,Mixer 14,Sports,279.24
515,Novel 15,Kitchen,212.48
516,Tennis Racket 16,Sports,463.41
517,Novel 17,Electronics,158.77
518,Phone 18,Books,355.25
519,Smartwatch 19,Kitchen,91.74
520,Headphones 20,Home,45.79
521,Laptop 21,Sports,134.42
522,Laptop 22,Sports,68.65
523,Headphones 23,Home,104.62
524,Headphones 24,Kitchen,72.61
525,Vacuum Cleaner 25,Sports,41.93
526,Camera 26,Kitchen,438.17
527,Smartwatch 27,Books,323.7
528,Smartwatch 28,Books,183.76
529,Novel 29,Books,413.2
530,Novel 30,Sports,42.82
531,Blender 31,Electronics,344.85
532,Phone 32,Sports,276.18
533,Smartwatch 33,Home,439.83
534,Laptop 34,Kitchen,231.03
535,Vacuum Cleaner 35,Home,294.9
536,Blender 36,Sports,371.64
537,Vacuum Cleaner 37,Electronics,424.65
538,Laptop 38,Sports,356.38
539,Smartwatch 39,Electronics,471.43
540,Smartwatch 40,Sports,442.9
541,Smartwatch 41,Sports,91.21
542,Blender 42,Sports,285.61
543,Smartwatch 43,Sports,471.68
544,Vacuum Cleaner 44,Books,81.45
545,Tennis Racket 45,Sports,62.0
546,Mixer 46,Sports,425.46
547,Novel 47,Home,343.14
548,Tennis Racket 48,Electronics,154.69
549,Phone 49,Electronics,132.86
550,Headphones 50,Home,376.59
```

```sql
DROP TABLE IF EXISTS products; 

CREATE TABLE products ( 
    product_id INT PRIMARY KEY, 
    product_name VARCHAR(100), 
    category VARCHAR(50), 
    price NUMERIC(10,2) 
); 
``` 
```sql
DROP TABLE IF EXISTS customers; 

CREATE TABLE customers ( 
customer_id INT PRIMARY KEY, 
customer_name VARCHAR(100), 
region VARCHAR(50), 
signup_date DATE 
);  
```
```sql
DROP TABLE IF EXISTS orders; 

CREATE TABLE orders (
	order_id INT PRIMARY KEY, 
	customer_id INT, 
	product_id INT, 
	quantity INT, 
	sale_date DATE, 
	amount NUMERIC(10,2),

	CONSTRAINT fk_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id),

    CONSTRAINT fk_product
        FOREIGN KEY (product_id)
        REFERENCES products(product_id)
);
```
```sql
SELECT column_name, data_type, is_nullable, character_maximum_length
FROM information_schema.columns
WHERE table_name = 'orders';
```
```sql
SELECT 
    CURRENT_DATE - sale_date AS days_diff
FROM sales_data;

select ceil(price) as ceilp, floor(price) as floorp 
from products;

select substring(product_name, 1, 5)
from products;

select concat(product_name, '-', category)
from products;

select signup_date,
extract(YEAR from signup_date) as years,
extract(MONTH from signup_date) as months,
extract(DAY from signup_date) as days
from customers;

select CAST(sale_date as text), sale_date as abcd
from orders;

select COALESCE(amount, quantity*10) as amt
from orders;

select amount,
	case
		when amount >= 500 then 'High Value'
		when amount >= 200 then 'Medium Value'
		when amount > 0 then 'Low Value'
		else 'No AMOUNT'
	end as sale_category
from orders;

select * from customers;

select customer_name, region, signup_date,
	case
		when region in ('North', 'South') then
			case
			    when extract(YEAR from signup_date)<'2022'  then 'Old'
		        else 'New'
			end
		else 'Other Region'
	end as type_customer
from customers;
```

### 1. Show each order and calculate **how many days ago** it was placed.

```sql
SELECT 
    order_id,
    sale_date,
    CURRENT_DATE - sale_date AS days_ago
FROM orders;
``` 
### 2. Find customers who signed up more than **30 days ago**.

```sql
SELECT *
FROM customers
WHERE signup_date < CURRENT_DATE - INTERVAL '30 days';
``` 
### 3. Classify orders based on amount:

* Above 1000 → "High Value"
* Between 500–1000 → "Medium Value"
* Otherwise:
  * If quantity > 2 → "Low but Bulk"
  * Else → "Low Value"

```sql
SELECT 
    order_id,
    amount,
    quantity,
    CASE
        WHEN amount > 1000 THEN 'High Value'
        WHEN amount BETWEEN 500 AND 1000 THEN 'Medium Value'
        ELSE 
            CASE 
                WHEN quantity > 2 THEN 'Low but Bulk'
                ELSE 'Low Value'
            END
    END AS order_class
FROM orders;
``` 
### 4. Classify customers by region:

* North → "Prime Region"
* South or East → "Medium Region"
* Else:

  * If name starts with A–M → "Group 1"
  * Otherwise → "Group 2"

```sql
SELECT 
    customer_id,
    customer_name,
    region,
    CASE
        WHEN region = 'North' THEN 'Prime Region'
        WHEN region IN ('South', 'East') THEN 'Medium Region'
        ELSE 
            CASE 
                WHEN SUBSTRING(customer_name FROM 1 FOR 1) BETWEEN 'A' AND 'M'
                    THEN 'Group 1'
                ELSE 'Group 2'
            END
    END AS region_category
FROM customers;
``` 
### 5. Show order amount rounded, and also show it rounded up and down.

```sql
SELECT
    order_id,
    amount,
    CEIL(amount) AS amount_up,
    FLOOR(amount) AS amount_down,
    ROUND(amount, 0) AS amount_rounded
FROM orders;
``` 
### 6. Display only the first 3 letters of each customer’s region.

```sql
SELECT 
    customer_id,
    region,
    SUBSTRING(region FROM 1 FOR 3) AS short_region
FROM customers;
``` 
### 7. Extract first 2 letters of the product category.

```sql
SELECT 
    product_id,
    product_name,
    SUBSTRING(category FROM 1 FOR 2) AS cat_code
FROM products;
``` 
### 8. Display "CustomerName (CustomerID)" for each customer.

```sql
SELECT 
    CONCAT(customer_name, ' (', customer_id, ')') AS customer_label
FROM customers;
``` 
### 9. Create a formatted text: `Customer: <name> - Region: <region>`

```sql
SELECT 
    CONCAT('Customer: ', customer_name, ' - Region: ', region) AS info
FROM customers;
``` 
### 10. Show total sales per month.

```sql
SELECT 
    EXTRACT(MONTH FROM sale_date) AS month,
    SUM(amount) AS total_sales
FROM orders
GROUP BY month
ORDER BY month;
```
### 11. Show how many orders were created per **year**.

```sql
SELECT 
    EXTRACT(YEAR FROM sale_date) AS year,
    COUNT(*) AS orders_in_year
FROM orders
GROUP BY year;
```  
### 12. Convert sale_date to text and amount to integer.

```sql
SELECT 
    order_id,
    CAST(sale_date AS TEXT) AS date_text,
    CAST(amount AS INT) AS amount_integer
FROM orders;
``` 
### 13. If amount is NULL (missing), display 0 instead.

```sql
SELECT 
    order_id,
    COALESCE(amount, 0) AS fixed_amount
FROM orders;
``` 
### 14. Replace missing customer names with "No Name".

```sql
SELECT
    customer_id,
    COALESCE(customer_name, 'No Name') AS fixed_name
FROM customers;
``` 
### 15. Create a label: "Order #ID: $Amount"

```sql
SELECT
    CONCAT('Order #', order_id, ': $', CAST(amount AS TEXT)) AS order_label
FROM orders;
``` 
### 16. Show customer initials and region, using "Unknown" if name missing.

```sql
SELECT
    CONCAT(
        SUBSTRING(COALESCE(customer_name, 'Unknown') FROM 1 FOR 1),
        '.',
        region
    ) AS customer_initial_and_region
FROM customers;
``` 
### 17. CONCAT + SUBSTRING + CAST** Show a product label like:
`P<product_id> - <first 4 letters of name>`

```sql
SELECT 
    CONCAT(
        'P', CAST(product_id AS TEXT), ' - ', SUBSTRING(product_name FROM 1 FOR 4)
    ) AS product_label
FROM products;
``` 
### 18. Classify orders placed within the last 7 days as "Recent", within the same month as "This Month", otherwise "Old".

```sql
SELECT 
    order_id,
    sale_date,
    CASE 
        WHEN sale_date >= CURRENT_DATE - INTERVAL '7 days' THEN 'Recent'
        WHEN EXTRACT(MONTH FROM sale_date) = EXTRACT(MONTH FROM CURRENT_DATE)
             AND EXTRACT(YEAR FROM sale_date) = EXTRACT(YEAR FROM CURRENT_DATE)
             THEN 'This Month'
        ELSE 'Old'
    END AS order_age
FROM orders;
``` 
### 19. Label orders based on age:

* Less than 3 days → "Very Recent"
* Less than 10 days → "Recent"
* Otherwise → "Old"

```sql
SELECT 
    order_id,
    sale_date,
    CASE
        WHEN sale_date >= CURRENT_DATE - INTERVAL '3 days' THEN 'Very Recent'
        WHEN sale_date >= CURRENT_DATE - INTERVAL '10 days' THEN 'Recent'
        ELSE 'Old'
    END AS date_status
FROM orders;
``` 
### 20. ROUND + EXTRACT** Show average order amount per day of the week (rounded to 1 decimal).

```sql
SELECT 
    EXTRACT(DOW FROM sale_date) AS weekday,
    ROUND(AVG(amount), 1) AS avg_amount
FROM orders
GROUP BY weekday;
``` 
### 21. COALESCE + CEIL** Use CEIL on amount but replace NULLs with 0.

```sql
SELECT 
    order_id,
    CEIL(COALESCE(amount, 0)) AS fixed_amount_up
FROM orders;
```