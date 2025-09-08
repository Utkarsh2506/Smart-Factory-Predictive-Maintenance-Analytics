CREATE DATABASE IF NOT EXISTS factory_db;
USE FACTORY_DB;

CREATE TABLE IF NOT EXISTS machine_data (
    ProductID VARCHAR(10),
    Type VARCHAR(1),
    AirTemperature DECIMAL(5,1),
    ProcessTemperature DECIMAL(5,1),
    RotationalSpeed INT,
    Torque DECIMAL(5,1),
    ToolWear INT,
    Target INT,
    FailureType VARCHAR(30)
);

SELECT 
    *
FROM
    FACTORY_DB.MACHINE_DATA
LIMIT 10;