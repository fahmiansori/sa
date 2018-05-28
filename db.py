import pymysql
import pandas as pd

class Database():
    def __init__(self):
        self.con = None

    def diconnect(self):
        self.con.close();
        print("Disconnected")
        return True

    def connect(self,host,user,password,db):
        success = False
        msg = ""
        if not host:
            msg = "No host. "
        if not user:
            msg += "No username. "
        if not db:
            msg += "No database. "
        if not password:
            password = ""

        if host and user and db:
            msg = ""
            try:
                self.con = pymysql.connect(host=host, user=user, passwd=password, database=db,charset='utf8')
                success = True
                msg = "Connected"
            except pymysql.err.InternalError as e:
                code,msg = e.args
                print("Error : ",msg)
            except pymysql.MySQLError as e:
                code,msg = e.args
                print("Error : ",msg)
            except:
                msg = "Unknow error. Failed to connect to database! Check your connection"
                # print("Failed to connect to database!")

        # ret = {"connection":self.con,"success":success}
        ret = {"success":success,"msg":msg}
        return ret

    def tables(self,db):
        query = "SELECT table_name FROM information_schema.tables where table_schema='{0}'"
        cursor = self.con.cursor()
        tables = None
        try:
            cursor.execute(query.format(db))
            rows = cursor.fetchall()
            if cursor.rowcount > 0:
                tables = rows
        except pymysql.MySQLError as e:
            print("Error retrieving tables")
            print('Got error {!r}, errno is {}'.format(e, e.args[0]))

        return tables

    def query(self,query):
        rows = None
        if self.con != None:
            cursor = self.con.cursor()
            rows = None
            try:
                cursor.execute(query)
                rows = cursor.fetchall()
            except pymysql.MySQLError as e:
                # print("Error retrieving tables")
                print('Got error {!r}, errno is {}'.format(e, e.args[0]))

        return rows

    def queryWithError(self,query):
        rows = None
        errorcode = 0
        if self.con != None:
            cursor = self.con.cursor()
            rows = None
            try:
                cursor.execute(query)
                rows = cursor.fetchall()
            except pymysql.MySQLError as e:
                # print("Error retrieving tables")
                errorcode = e.args[0]
                # print('Got error {!r}, errno is {}'.format(e, e.args[0]))

        ret = {}
        ret['rows'] = rows
        ret['errorcode'] = errorcode
        return ret

    def queryInsert(self,query):
        if self.con != None:
            cursor = self.con.cursor()
            try:
                cursor.execute(query)
                self.con.commit()
                # con.close()
            except pymysql.MySQLError as e:
                print("Error retrieving tables")
                print('Got error {!r}, errno is {}'.format(e, e.args[0]))

    def getDataAsDF(self,table,col='*',cond=''):
        if self.con != None:
            query = "SELECT {0} from {1} {2}"
            return pd.read_sql(query.format(col,table,cond), con=self.con)
        return None
