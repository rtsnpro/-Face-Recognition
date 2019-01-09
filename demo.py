import xlrd
import MySQLdb
# Open the workbook and define the worksheet
book = xlrd.open_workbook("signin.xls")
sheet = book.sheet_by_name("signin_now")

#創建一個MySQL連接
database = MySQLdb.connect (host="localhost", user = "root", passwd = "0000", db = "testsi")

# 獲得遊標對象, 用於逐行遍歷數據庫數據
cursor = database.cursor()

# 創建插入SQL語句
query = """INSERT INTO orders (ID, name, arrive) VALUES (%s, %s, %s)"""

# 創建一個for循環迭代讀取xls文檔每行數據的, 從第二行開始是要跳過標題
for r in range(1, sheet.nrows):
      ID        = sheet.cell(r,0).value
      name      = sheet.cell(r,1).value
      arrive    = sheet.cell(r,2).value


      values = (ID, name, arrive)

      # 執行sql語句
      cursor.execute(query, values)

# 關閉遊標
cursor.close()

# 提交
database.commit()

# 關閉數據庫連接
database.close()














"""
#encoding=utf-8
import xlrd
import mysqldb

data=http://www.xuebuyuan.com/xlrd.open_workbook('test.xlsx')
table=data.sheets()[0]

nrows=table.nrows
ncols=table.ncols

tabledict={}

for i in range(nrows):
   tabledict[i]=table.row_values(i)
   
print tabledict[2]
print tuple(tabledict[2])


#讀取數據
try:
   conn=MySQLdb.connect(host='localhost',user='root',passwd='1234',db='test',port=3306,charset='utf8')
   cur=conn.cursor()
   cur.execute('select Name, sex from classmate')
   result_set=cur.fetchall()
   for row in result_set:
      print row
   print "Number of rows returned: %d"%cur.rowcount
   cur.close()
   conn.close()
except MySQLdb.Error,e:
   print "MySQL Error %d:%s"%(e.args[0],e.args[1])
   
#插入數據
try:
   conn=MySQLdb.connect(host='localhost',user='root',passwd='1234',db='test',port=3306,charset='utf8')
   cur=conn.cursor()
   sql1="DROP table IF EXISTS ExcelTable;"
   cur.execute(sql1)
   print "Drop success!"
   sql2="create table IF NOT EXISTS ExcelTable(col1 varchar(20) primary key, col2 varchar(256),col3 int(10))"
   cur.execute(sql2)
   print "Sucess to create a new table!"
   
   #列表轉元組，tabledict[i]
   
   #通過添加數據到列表中，然後再轉為元組，因為元組是不可改的。
   sql3="insert into ExcelTable (col1,col2,col3) values(%s,%s,%s)"
   param01=[]
   for i in range(nrows):
      param01.append(tuple(tabledict[i]))
   param02=tuple(param01)
   print param02
   
   #多行數據
   try:
      cur.executemany(sql3,param02)
      conn.commit()
      print "success insert many records!"
   except Exception,e:
      conn.rollback()
      print e
   
   cur.close()
   conn.close()
except MySQLdb.Error,e:
   print "MySQL Error %d:%s"%(e.args[0],e.args[1])

"""
"""
功能：將Excel數據導入到MySQL數據庫
"""

