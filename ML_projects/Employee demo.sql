SELECT * FROM school.employee;
alter table employee add age int after name;
update employee set age =case
when empid =1 then 28
when empid=2 then 32
when empid =3 then 44
when empid=4 then 29
when empid=5 then 29
when empid=6 then 45
when empid=7 then 23
end;