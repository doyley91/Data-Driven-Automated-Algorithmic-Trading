CREATE DATABASE securities_master;

CREATE USER sec_user WITH PASSWORD 'Password';
GRANT ALL PRIVILEGES ON DATABASE securities_master to sec_user;

CREATE TABLE exchange (
    id int PRIMARY KEY NOT NULL,
    abbrev varchar(32) NOT NULL,
    name varchar(255) NOT NULL,
    city varchar(255) NULL,
    country varchar(255) NULL,
    currency varchar(64) NULL,
    timezone_offset time NULL,
    created_date timestamp NOT NULL,
    last_updated_date timestamp NOT NULL
);

CREATE TABLE data_vendor (
    id int PRIMARY KEY NOT NULL,
    name varchar(64) NOT NULL,
    website_url varchar(255) NULL,
    support_email varchar(255) NULL,
    created_date timestamp NOT NULL,
    last_updated_date timestamp NOT NULL
);

CREATE TABLE symbol (
  id int PRIMARY KEY NOT NULL,
  ticker varchar(32) NOT NULL,
  instrument varchar(64) NOT NULL,
  name varchar(255) NULL,
  sector varchar(255) NULL,
  currency varchar(32) NULL,
  created_date timestamp NOT NULL,
  last_updated_date timestamp NOT NULL,
  exchange_id INT REFERENCES exchange (id)
);

CREATE TABLE daily_price (
  id int PRIMARY KEY NOT NULL,
  price_date timestamp NOT NULL,
  created_date timestamp NOT NULL,
  last_updated_date timestamp NOT NULL,
  open_price decimal(19,4) NULL,
  high_price decimal(19,4) NULL,
  low_price decimal(19,4) NULL,
  close_price decimal(19,4) NULL,
  volume bigint NULL,
  adj_open_price decimal(19,4) NULL,
  adj_high_price decimal(19,4) NULL,
  adj_low_price decimal(19,4) NULL,
  adj_close_price decimal(19,4) NULL,
  adj_volume bigint NULL,
  ex-dividend decimal(19,4) NULL,
  split_ratio decimal(19,4) NULL,
  data_vendor_id INT REFERENCES data_vendor (id),
  synbol_id INT REFERENCES symbol (id)
);