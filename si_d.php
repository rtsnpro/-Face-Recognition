<?php
require_once "config.php";

$db = new mysqli($hostname, $user, $passwd, $database);
$db->set_charset("utf-8");
$file= fopen("signin.txt","r");
$content = fgets($file);

if(isset($_POST['name']) or isset($_POST['sitime_m'])){
  $name=$_POST['name'];
  $sitime_m=$_POST['sitime_m'];
  $sql= "select * from si_now where name like '%$name%' and sitime_m like '%$sitime_m%' "; 
 }else{
  $name='';
  $sitime_m=$content;
  $sql= "select * from si_now where sitime_m like '%$sitime_m%'";
 }

#$sql = "select * from si_now";

$stmt = $db->prepare($sql);
$stmt->execute();

$result = $stmt->get_result();
$rows = [];

while ($row = $result->fetch_array())
{
    $rows[] = $row;
}

$stmt->close();
$db->close();
?>

<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
  <meta name="description" content="">
  <meta name="author" content="templatemo">

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

        












  <!--<h3>辨識出席結果</h3>-->


    <h3 align='center'><form  id="form1" name="form1" method="post" action="" > &nbsp; &nbsp;依據條件搜尋——
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;姓名：
        <input name="name" type="text" id="name" value="<?php echo $name?>" />
        &nbsp;&nbsp;&nbsp;&nbsp;簽到時間：
        <input name="sitime_m" type="text" id="sitime_m" value="<?php echo $sitime_m?>" />
      
  
      <!-- <p>
       <form>
          <select name="YourLocation">
        
　           <option value="Taipei">台北</option>
　           <option value="Taoyuan">桃園</option>
　           <option value="Hsinchu">新竹</option>
　           <option value="Miaoli">苗栗</option>
             

          </select>
        </form>
      </p> -->
      
      &nbsp;&nbsp;&nbsp;&nbsp;<input type="submit" name="button" id="button" value="搜尋" />
      
    </form></h3>
  
  
<section class="templatemo-gray-bg">
  <table width=60% align='center'>
    <tr>
      <th align='center' bgcolor=#DDAA55 ><font size="5"><b>編號</b></font></th>
      <th align='center' bgcolor=#DDAA55 ><font size="5"><b>姓名</b></font></th>
      <th align='center' bgcolor=#DDAA55 ><font size="5"><b>出席狀態</b></font></th>
      <th align='center' bgcolor=#DDAA55 ><font size="5"><b>出席時間</b></font></th>
      <th align='center' bgcolor=#DDAA55 ><font size="5"><b>簽到時間</b></font></th>
    </tr>

    <?php
    
    foreach ($rows as $row)
    {
        echo "<tr>";
        echo "<td align='center' bgcolor=#FFF5EE><font size='4'><b>".$row['id']."</b></font></td>";
        echo "<td align='center' bgcolor=#FFF5EE><font size='4'><b>".$row['name']."</b></font></td>";
        echo "<td align='center' bgcolor=#FFF5EE><font size='4'><b>".$row['arrive']."</b></font></td>";
        echo "<td align='center' bgcolor=#FFF5EE><font size='4'><b>".$row['sitime_p']."</b></font></td>";
        echo "<td align='center' bgcolor=#FFF5EE><font size='4'><b>".$row['sitime_m']."</b></font></td>";
        echo "</tr>";
    }
    
    ?>
  </table> 

  <p>&nbsp;</p>
  </section>
  <footer class="text-center">
            <p class="text-uppercase">
            	
            </p>
        </footer>



</body>

</html>