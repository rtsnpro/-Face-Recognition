SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";

CREATE DATABASE IF NOT EXISTS Levels DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
use signin;

CREATE TABLE `si_now` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(32) NOT NULL COMMENT '姓名',
  `arrive` varchar(8) NOT NULL COMMENT '出席',
  `sitime_p` varchar(20) NOT NULL COMMENT '出席時間',
  `sitime_m` varchar(20) NOT NULL COMMENT '簽到日期',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;


CREATE TABLE `si_times` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `sitime_m` varchar(20) NOT NULL COMMENT '簽到日期',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;