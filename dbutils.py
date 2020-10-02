__doc__ = """
dbutils connects the python application to a MS Sql Server DB
=============================================================

The functionality revolves around the class "Dbconn()" which, once created, holds
the connection to the DB. You can then read/write data to/from the DB using methods
of this object.

The connection details e.g. the server name and driver are hardcoded as a class
variable here so that if there are ever any migrations or any changes in the DB
you wil just have to make the one change here and not everywhere in the code.

There is a SQLBuilder() class which helps build out SQL queries for the user
who does not know how to write SQL queries. Running this module as a script
will show a small worked example of how exactly this class can be used to write
a SQL query. Code is at the very bottom of the module.
"""

import pyodbc
import pandas as pd
import datetime
from datetime import timedelta
import numpy as np

db_name = "fsiaufi"
db_schema = 'dbo'

class Dbconn():
    """ DB Connector Object. """
    def __init__(self):
        db_driver = "{ODBC Driver 13 for SQL Server}"
        db_server = "SYD4-TD-RIS01.FIRSTSTATE.FSIORG.COM"
        # db_server = "N010NRIW3525.aud01.cbaidev01.com"
        db_trusted = "yes"
        try:
            self.connection = pyodbc.connect(
                "Driver={};"
                "Server={};"
                "Database={};"
                "Trusted_Connection={};".format(db_driver, db_server, db_name, db_trusted)
            )
            # print("Connection to SQL Server DB successful")
            self.connection.autocommit = True
        except Exception as e:
            print(f"The error '{e}' occurred")
            self._write_err_to_log(f"The error '{e}' occurred")
            self.conn_success = False
    
    def _write_err_to_log(self,s):
        """ Writes error to a .txt. file """
        f = open('saveprogress.txt','a+')
        f.write(f'{s}\r\n')
        f.close()
    
    # This will also take a tuple
    def _list_to_sql_friendly_str(self, in_tuple):
        ''' This has been replaced with the _force_speechmarks() function and should
        be deleted from the source files along with any references. '''
        out = "("
        for i, item in enumerate(in_tuple,1):
            if isinstance(item,str):
                if "'" in item:
                    item = item.replace("'","''")
                out = out + "\'" + item + "\'"
            elif isinstance(item,datetime.date):
                out = out + "\'" + item.strftime('%Y-%m-%d') + "\'"
            elif isinstance(item,float) or isinstance(item,np.integer):
                out = out + str(item)
            else:
                out = out + item
            if i != len(in_tuple):
                out = out + ","
        return out + ")" 

    # This is best used for the SELECT portion of the statement as you will want double quotes
    def _force_speechmarks(self, in_list, add_brackets = True, single_quotes = False, keep_dates = True):
        """
        Fore either single speechmarks or double speechmarks around strings or lists of strings/numbers/dates
        Note: You will want to use Double Quotes for field names and Single Quotes for field values. This is the 
        MS Sql Server convention which you must adhere to and what makes this function invaluable.

        Parameters
        ----------
        in_list : list or string
            Pass a string which you wish to manipulate, or a list of strings/doubles/dates which you wish to separate with commas too.
        add_brackets : Bool, optional
            True will surround the string or all the comma separated values produced from a list with parenthesis. The default is True.
        single_quotes : Bool, optional
            Force single quotes. The Default is double quotes. The default is False.
        keep_dates : Bool, optional
            Keep dates formatted as date objects. Else will force to string. The default is True.

        Returns
        -------
        out : string
            A string manipulated by the brackets/quotes you have requested. Ready to be injected into a SQL query fstring.

        """
        out = ""
        if(add_brackets):
            out = "("
        if isinstance(in_list, datetime.date):
            in_list = in_list.strftime('%Y-%m-%d')
            if(single_quotes):
                out = out + "\'" + str(in_list) + "\'"
            else:
                out = out + "\"" + str(in_list) + "\""
        elif isinstance(in_list, str):
            if isinstance(in_list,datetime.date):
                in_list = in_list.strftime('%Y-%m-%d')
            if(single_quotes):
                out = out + "\'" + str(in_list) + "\'"
            else:
                out = out + "\"" + str(in_list) + "\""
        else:
            for i, item in enumerate(in_list,1):
                if isinstance(item,datetime.date):
                    item = item.strftime('%Y-%m-%d')
                if(single_quotes):
                    out = out + "\'" + str(item) + "\'"
                else:
                    out = out + "\"" + str(item) + "\""
                if i != len(in_list):
                    out = out + ","
        if(add_brackets):
            out = out + ")"
        return out

    def db_save(self, table_name, table_headers, row_values):
        """
        Save values to a DB

        Parameters
        ----------
        table_name : String
            The database table name.
        table_headers : list
            List of strings corresponding to the table fields you are saving to.
        row_values : list
            list of values corresponding to the table fields you want to save down.

        Returns
        -------
        None.

        """        
        try:
            cur = self.connection.cursor()
            cur.execute("INSERT INTO {3}.dbo.{0} {1} VALUES {2}".format(self._force_speechmarks(table_name,False),
                self._force_speechmarks(table_headers), self._list_to_sql_friendly_str(row_values), db_name))
            cur.close()
        except Exception as e:
            print(f"Did NOT save to DB [{e}]")
            print(row_values)
            self._write_err_to_log(f"Did NOT save to DB [{e}]")
            self._write_err_to_log(str(row_values))

    def db_read(self, table_name, query_from): 
        """
        Read values from the DB without having to write out your own SQL query.

        Parameters
        ----------
        table_name : String
            The name of the table in the database.
        query_from : string or List
            The field names you wish to pull from the DB. Separate numerous values in a list if you want many fields.

        Returns
        -------
        Pandas.DataFrame
            DataFrame with the requested output.

        """
        query_from = self._force_speechmarks(query_from, False, False)
        sql = "SELECT {0} FROM {2}.dbo.{1}".format(query_from, table_name, db_name)
        return pd.read_sql(sql,self.connection)

    def db_run_sql(self, sql):
        ''' Run a SQL query you have written and passed as a variable '''
        return pd.read_sql(sql,self.connection)

    def db_read_where(self, table_name, fields_to_pull, field_to_match, value_to_match, force_type = ''):
        """
        Match a value on a table field and return the value of another field.

        Parameters
        ----------
        table_name : String
            Table name.
        fields_to_pull : String or List
            String or list of many strings for fields you wish to pull.
        field_to_match : String
            The field which you are filtering on.
        value_to_match : Variable
            The value you are filtering the field_to_match on.
        force_type : string, optional
            Force the output type. Can force to list if there is a | delimeter and a dict if there are : delimters.
            This is used primarily to parse values from the GenKeyVals table and probably not fit for general use. The default is ''.

        Returns
        -------
        Variable
            Returns if any data is found for your filter.

        """
        cur = self.connection.cursor()
        cur.execute( "SELECT {0} FROM {4}.dbo.{1} WHERE {2} = {3}".format(self._force_speechmarks(fields_to_pull,False),
            self._force_speechmarks(table_name,False),self._force_speechmarks(field_to_match,False),
            self._force_speechmarks(value_to_match,False,True),
            db_name))
        read_res = cur.fetchall()
        cur.close()
        if force_type.lower() == '':
            return read_res
        if force_type.lower() == 'list':
            return str.split(read_res[0][0],"|")
        if force_type.lower() == 'dict':
            return dict(item.split(":") for item in read_res[0][0].split("|"))
        return read_res
    
    def get_colnames_list(self, table_name):
        ''' Returns column names for a table_name supplied '''
        res = self.db_run_sql('SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = {}'.format(
            self._force_speechmarks(table_name,False,True)))
        unq = res['COLUMN_NAME'].tolist()
        if 'NumID' in unq: unq.remove('NumID')
        return unq
    
    def sqlbuilder_run(self,table_name, df):
        """
        Takes a df in from the sqlbuilder gui tool and concats WHERE clauses.

        Parameters
        ----------
        table_name : String
            The table name.
        df : Pandas.DataFrame()
            The DataFrame which is supplied from the GUI which outlines the filters to apply.

        Returns
        -------
        Pandas.DataFrame
            If there is anything found for the query you have built, it will returned a df with that data.

        """
        sql = 'SELECT * FROM {}.dbo.{} WHERE '.format(
            db_name,
            self._force_speechmarks(table_name,False))
        for index, row in df.iterrows():
            if index != 0:
                sql += '{} '.format(row['Conjunction'].upper())
            sql += '{} {} {} '.format(self._force_speechmarks(row['Field'],False),
                                     row['Operation'].upper(),
                                     self._force_speechmarks(row['Value'],False,True))
        return self.db_run_sql(sql)

    def get_distinct_list(self, table_name, col_name):
        """
        Returns a list of distinct values from a table and column

        Parameters
        ----------
        table_name : String
            Table name.
        col_name : String
            Column name.

        Returns
        -------
        list
            list of distinct values.

        """
        if col_name != '':
            res = self.db_run_sql('SELECT DISTINCT {} FROM {}.dbo.{}'.format(
                self._force_speechmarks(col_name,False),
                db_name,
                self._force_speechmarks(table_name,False)))
            res = res[col_name].tolist()
            return list(map(str,res))
    
    def prev_weekday(self, adate, n_days):
        ''' Find a weekday n_days before a date adate. '''
        adate -= timedelta(days=n_days)
        while adate.weekday() > 4: # Mon-Fri are 0-4
            adate -= timedelta(days=1)
        return adate

    def __del__(self):
        try:
            self.connection.close()
        except AttributeError:
            pass

class SQLBuilder(Dbconn):
    ''' SQLBuilder is a class which helps build complex SQL queries despite the user not knowing
    the SQL language. When creating the class, you must add the below two parameters.
    
    You can add as many where clauses as you want by using the add_where_filter method.
        
    Parameters
    ----------
    table_name : String
        Table name.
    fields_to_pull : String or list
        Column name.'''
    def __init__(self, table, fields_to_pull = None ):
        self.table = table
        self.select = fields_to_pull
        self.where = []
    
    def set_table(self, table):
        self.table = table
    
    def order_by(self, order_by, descending = False):
        self.order = order_by
        self.order_desc = descending
    
    def build(self):
        ''' Builds and returns the SQL query. This can then be plugged into a function which
        can run this query for you. '''
        sql = ''
        if self.select is None:
            sql = 'SELECT *'
        else:
            sql = 'SELECT {}'.format(super()._force_speechmarks(self.select, False, False))
        sql += ' FROM {}.{}.{}'.format(
            db_name, db_schema, self.table
            )
        
        # WHERE Clause
        if len(self.where) > 0:
            sql += ' WHERE '
            for i, item in enumerate(self.where):
                if i > 0:
                    sql += ' ' + item.conj + ' '
                if item.start_parenth:
                    sql += '('
                sql += '{} {} {}'.format(
                    super()._force_speechmarks(item.field, False, False),
                    item.operation,
                    super()._force_speechmarks(item.value, True, True)
                    )
                if item.end_parenth:
                    sql += ')'
        
        # ORDER Clause
        if self.order is not None:
            sql += ' ORDER BY {}'.format(self._force_speechmarks(self.order,False,False))
            if self.order_desc:
                sql += ' DESC'

        return sql
    
    def add_where_filter(self, conjunction, field, operation, value, start_parenthesis = False, end_parenthesis = False):
        """
        Adds a where filter to your SQLBuilder object. You can define speific filters and as many as you please by calling this function
        repeatedly on your SQLBuilder object. You can also open and close parentheses on your filters here so you can build more nuanced
        queries. Pertinent info: The WHERE filters you create will be called in order (this is essential in order for the open/close bracket)
        functionalities to work.

        Parameters
        ----------
        conjunction : String
            Choose between "AND" or "OR" This is only important beyond the first call of this member. The first where clause is unaffected by this.
        field : String
            The field name which you wish to filter.
        operation : String
            Choose between "=", "<", ">", "IN", "NOT IN".
        value : String or List
            Must be String for the "=", "<", ">" else you can use lists for the "IN", "NOT IN" operators.
        start_parenthesis : Bool, optional
            Opens the parenthesis for your where clause if required. The default is False.
        end_parenthesis : Bool, optional
            Closes the parenthesis for your where clause. Only necessary if you have opened a parenthesis previously. The default is False.

        Returns
        -------
        None.

        """
        self.where.append(self.Where(conjunction, field, operation, value, start_parenthesis, end_parenthesis))
    
    class Where():
        """ Where objet holds the variables required for the SQLBuilder to build out filtered queries """
        def __init__(self, conjunction, field, operation, value, start_parenthesis = False, end_parenthesis = False):
            self.conj = conjunction
            self.field = field
            self.operation = operation
            self.value = value
            self.start_parenth = start_parenthesis
            self.end_parenth = end_parenthesis

if __name__ == '__main__':    
    testlist = ['Date', 'Country', 'CurveName']
    sqlobj = SQLBuilder('RVCoeffs')
    sqlobj.add_where_filter(None, 'Country', '=', 'AU')
    sqlobj.add_where_filter('AND', 'CurveName', 'IN', ['Treasury','Utilities','Corporates'], True)
    sqlobj.add_where_filter('AND', 'Date','>','2020-08-01',False,True)
    sqlobj.order_by(['Date','CurveName'])
    print(sqlobj.build())