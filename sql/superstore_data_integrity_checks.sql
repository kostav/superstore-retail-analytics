use sakila;

-- Step 0A: Superstore Data Integrity & Cleaning Checks

-- 1) Checking for Nulls in critical columns
SELECT 
  SUM(CASE WHEN "Order Date" IS NULL THEN 1 ELSE 0 END) AS null_order_date,
  SUM(CASE WHEN "Region" IS NULL THEN 1 ELSE 0 END) AS null_region,
  SUM(CASE WHEN "Category" IS NULL THEN 1 ELSE 0 END) AS null_category,
  SUM(CASE WHEN "Sub-Category" IS NULL THEN 1 ELSE 0 END) AS null_subcategory,
  SUM(CASE WHEN "Sales" IS NULL THEN 1 ELSE 0 END) AS null_sales,
  SUM(CASE WHEN "Profit" IS NULL THEN 1 ELSE 0 END) AS null_profit,
  SUM(CASE WHEN "Discount" IS NULL THEN 1 ELSE 0 END) AS null_discount,
  SUM(CASE WHEN "Customer Segment" IS NULL THEN 1 ELSE 0 END) AS null_segment
FROM superstore;

-- 2) Checking for Invalid discount range
SELECT COUNT(*) AS invalid_discount_rows
FROM superstore
WHERE "Discount" < 0 OR "Discount" > 1;

-- 3) Checking if there are potential duplicates (Row with same Order & Customer & Product ID), 
--    in order for an order to be considered duplicate all the criteria below must apply 
SELECT
  `Order ID`,
  `Customer ID`,
  `Product ID`,
  COUNT(*)              AS row_count,
  GROUP_CONCAT(`Quantity` ORDER BY `Quantity`)       AS row_qty,
  GROUP_CONCAT(`Sales`    ORDER BY `Sales`)          AS row_sales,
  GROUP_CONCAT(`Profit`   ORDER BY `Profit`)         AS row_profit,
  GROUP_CONCAT(`Discount` ORDER BY `Discount`)       AS row_dscount,
  GROUP_CONCAT(`Row ID`   ORDER BY `Row ID`)         AS row_ids
FROM superstore
GROUP BY `Order ID`,`Customer ID`,`Product ID`
HAVING COUNT(*) > 1
ORDER BY row_count DESC, `Order ID`, `Product ID`;

-- 4) Distinct values for categoricals to check if there are any spelling issues or wrong data in general
SELECT DISTINCT `Region` FROM superstore; 
SELECT DISTINCT `Category` FROM superstore;
SELECT DISTINCT `Sub-Category` FROM superstore;
SELECT DISTINCT `Segment` FROM superstore;
