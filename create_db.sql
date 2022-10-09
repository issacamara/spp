
CREATE SCHEMA `spp` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin ;

create table spp.stock_news
(
    id       int auto_increment
        primary key,
    ticker   varchar(10)   null,
    exchange varchar(15)   null,
    news     varchar(2000) null,
    date     datetime      null
);

SET SQL_SAFE_UPDATES = 0;

SET @@global.sql_mode= '';
