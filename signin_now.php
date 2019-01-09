<?php

require 'vendor/autoload.php';

include('config.php'); //連接數據庫
$db = new mysqli($hostname, $user, $passwd, $database);
$db->set_charset("utf-8");


#############上傳出席狀態給資料表si_now######

$reader = \PhpOffice\PhpSpreadsheet\IOFactory::createReader('Xls');
$reader->setReadDataOnly(TRUE);
$spreadsheet = $reader->load('signin.xls'); //載入excel表格

$worksheet = $spreadsheet->getActiveSheet();
$highestRow = $worksheet->getHighestRow(); // 總行數
$highestColumn = $worksheet->getHighestColumn(); // 總列數
$highestColumnIndex = \PhpOffice\PhpSpreadsheet\Cell\Coordinate::columnIndexFromString($highestColumn); // e.g. 5

$lines = $highestRow - 2; 
if ($lines <= 0) {
    exit('Excel表格中沒有數據');
}

$sql = "INSERT INTO `si_now` (`name`, `arrive`,`sitime_p`,`sitime_m`) VALUES ";

for ($row = 3; $row <= $highestRow; ++$row) {
    $name = $worksheet->getCellByColumnAndRow(1, $row)->getValue(); //姓名
    $arrive = $worksheet->getCellByColumnAndRow(2, $row)->getValue(); //出席
    $sitime_p = $worksheet->getCellByColumnAndRow(3, $row)->getValue(); //出席時間
    $sitime_m = $worksheet->getCellByColumnAndRow(4, $row)->getValue(); //簽到時間

    $sql .= "('$name','$arrive','$sitime_p','$sitime_m'),";
}
$sql = rtrim($sql, ","); //去掉最後一個,號
$db->query($sql);



#############上傳時間給資料表si_times######
$file= fopen("signin.txt","r");

while(!feof($file)){
    $content = fgets($file);
    $carray = explode(",", $content);
    list($sitime_m) = $carray;
    $sql2 ="INSERT INTO `si_times` (`sitime_m`) VALUES ('$sitime_m')";
    $db->query($sql2);
}  
fclose($file);

#echo '出席狀態上傳完畢';
?>


<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
  <meta name="description" content="Web Programming">

  <title>Sign in page</title>
  <style>
     table, tr, td, th
     {
        border: 1px solid black;
        border-collapse:collapse;
        padding: 5px; 
     }
  </style>



<link href="http://fonts.googleapis.com/css?family=Open+Sans:400,300,400italic,700" rel="stylesheet" type="text/css">
<link href="http://fonts.googleapis.com/css?family=Dancing+Script" rel="stylesheet" type="text/css">
<link href="css/font-awesome.min.css" rel="stylesheet">
<!--<link href="css/bootstrap.min.css" rel="stylesheet">-->
<link href="css/templatemo-style.css" rel="stylesheet">

</head>

<body>
        <!-- Header -->
        <div class="templatemo-container">
            <div class="templatemo-block-left">
                <div class="templatemo-header-left">
                    <div class="templatemo-header-text-wrap">
                        <div class="templatemo-header-text">
                            <h1 class="text-uppercase templatemo-site-name"><span class="gold-text"><b>人臉辨識</b></span> 點名系統</h1>
                        </div>
                    </div>
                    <div class="templatemo-header-left-overlay"></div>
                </div>
            </div>
            <div class="templatemo-block-right">
                <div class="templatemo-header-right">
                    <div class="templatemo-header-right-overlay"></div>
                </div>
            </div>
        </div> <!-- end Header -->




 <p align='center'><font size="5"><b>出席狀態上傳完畢 <a href="si_d.php"> [查看結果] </a></b></font></p>


   <footer class="text-center">
        <p class="text-uppercase">
            	
        </p>
    </footer>
</body>

</html>