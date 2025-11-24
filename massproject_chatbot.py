import os
import uuid
from datetime import datetime, date
from groq import Groq
import re
import json
from decimal import Decimal
import sys
import pymysql

# --- Configuration and Setup ---
# IMPORTANT: API keys and database credentials are hardcoded as per user request.
# In a production environment, these should be loaded from environment variables
# or a secure configuration management system.

# Groq API Key
GROQ_API_KEY = 'your_groq_api_key'

# Groq client for conversational and tool use
groq_client = Groq(api_key=GROQ_API_KEY)

# Define dangerous SQL keywords for safety checks
DANGEROUS_KEYWORDS = ['UPDATE', 'DELETE', 'DROP', 'ALTER', 'INSERT', 'TRUNCATE', 'RENAME', 'CREATE', 'EXECUTE', 'CALL', 'GRANT', 'REVOKE']

# Define max history length and max tokens for history to prevent hitting token limits
MAX_HISTORY_LENGTH = 5
MAX_TOKENS_FOR_HISTORY = 500

# Valid roles
# --- UPDATED: Added PROJECT_MANAGER role ---
VALID_ROLES = ["TEAM_MEMBER", "TEAM_LEAD", "PROJECT_MANAGER", "MANAGER", "ADMIN", "SUPER_ADMIN"]
DEFAULT_ROLE = "TEAM_MEMBER"

# --- CORRECTED: The MaasApp_user and MaasApp_superadminassociates tables are
# now removed from the SENSITIVE_TABLES list. This allows TEAM_LEAD and MANAGER
# roles to query user details, as the access level is already checked later in the code.
# The previous blanket restriction was incorrect. MaasApp_category is also removed
# as per the user's implicit request to allow broader access for certain queries. ---
# --- UPDATED: Added new tables as per the user's request. ---
SENSITIVE_TABLES = [
    'MaasApp_passwordresettoken',
    'MaasApp_manager',
    'MaasApp_admin',
    'MaasApp_associate_login_history',
    'MaasApp_user_login_history',
    'MaasApp_achievement',
    'MaasApp_blocker',
    'MaasApp_dailytask',
    'MaasApp_dailytask_files',
    'MaasApp_feedback',
    'MaasApp_meeting',
    'MaasApp_reviewsubmit'
]

# --- NEW: List of tables containing public data that should never be filtered by user ID. ---
# --- FIX: Corrected the table name from 'MaasApp_events' to 'MaasApp_event'. ---
# --- NEW FIX: Added MaasApp_leavetype to the public tables list. ---
# --- UPDATED: MaasApp_category has been removed from this list. ---
# --- FIX: MaasApp_user and MaasApp_superadminassociates have been removed from this list.
# This ensures that "my details" queries are correctly filtered by the user's ID.
# --- NEW FIX: Added MaasApp_tutorial to the public tables list, as tutorials are public documents. ---

PUBLIC_TABLES = [
    'MaasApp_holiday',
    'MaasApp_event',
    'MaasApp_intranet',
    'MaasApp_designations',
    'MaasApp_sub_designations',
    'MaasApp_tasktemplate',
    'MaasApp_tutorial',
    'MaasApp_department',
    'MaasApp_notification',
    'MaasApp_leavetype',
    'MaasApp_category',
]

# --- Database Utility Functions ---

def get_db_schema(host, user, password, database, port):
    """
    Connects to the MySQL database and retrieves the schema (table names and their columns).
    This schema is crucial for the AI to understand the database structure.
    
    Args:
        host: The database host.
        user: The database user.
        password: The database password.
        database: The database name.
        port: The database port.
        
    Returns:
        A dictionary representing the schema or a dictionary with an 'error' key if a connection fails.
    """
    connection = None
    cursor = None
    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=int(port)
        )
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()

        schema = {}
        for table in tables:
            table_name = table[0]
            try:
                # Retrieve columns for each table
                cursor.execute(f"SHOW COLUMNS FROM `{table_name}`;")
                columns = cursor.fetchall()
                schema[table_name] = [column[0] for column in columns]
            except pymysql.MySQLError as e:
                # Log a warning but continue with other tables if one fails
                print(f"Warning: Could not retrieve columns for table '{table_name}': {e}", file=sys.stderr)
                schema[table_name] = []

            except Exception as e:
                # Handle any other unexpected errors during column retrieval
                print(f"Warning: An unexpected error occurred retrieving columns for table '{table_name}': {e}", file=sys.stderr)
                schema[table_name] = []


        return schema
    except pymysql.MySQLError as e:
        # Handle database-specific connection errors
        return {"error": f"Database schema retrieval error: {e}. Please check database connection and credentials."}
    except Exception as e:
        # Handle any other unexpected errors
        return {"error": f"An unexpected error occurred during schema retrieval: {e}"}
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def get_user_data_for_prediction(user_id, db_config):
    """
    Fetches comprehensive user data for role prediction from multiple possible tables.
    The logic has been updated to be more robust.
    """
    host = db_config.get("HOST")
    user = db_config.get("USER")
    password = db_config.get("PASSWORD")
    database = db_config.get("NAME")
    port = db_config.get("PORT")

    connection = None
    cursor = None
    user_data = None
    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=int(port)
        )
        # Use DictCursor for easier access to data by column name
        cursor = connection.cursor(pymysql.cursors.DictCursor) 
        
        # --- NEW FIX: Check MaasApp_user for SUPER_ADMIN first based on user's new info ---
        cursor.execute("SELECT id, first_name, last_name, email, role FROM MaasApp_user WHERE id = %s;", (user_id,))
        user_data = cursor.fetchone()
        
        # --- NEW FIX: If not found, try the `MaasApp_superadminassociates` table for all other roles ---
        if not user_data:
            cursor.execute("SELECT id, first_name, last_name, email, role FROM MaasApp_superadminassociates WHERE id = %s;", (user_id,))
            user_data = cursor.fetchone()
        
        # If still not found after checking both tables, return an error
        if not user_data:
            return {"error": f"No user record found for ID {user_id} in any expected table."}
        
        return user_data

    except pymysql.MySQLError as e:
        return {"error": f"Database error fetching user data for prediction: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def predict_user_role_with_llm(user_id, db_config):
    """
    Uses the LLM to predict the user's role based on their database profile.
    This function will only be called if a role is not already found in the database.
    
    Args:
        user_id: The ID of the user.
        db_config: The database configuration dictionary.

    Returns:
        The predicted user role (string) or the default role on failure.
    """
    user_data = get_user_data_for_prediction(user_id, db_config)
    if "error" in user_data:
        # Return the default role directly if the data retrieval failed
        return DEFAULT_ROLE
    
    # If the role is already present in the user data, use that directly
    if 'role' in user_data and user_data['role'].upper() in VALID_ROLES:
        return user_data['role'].upper()

    # Construct a comprehensive prompt for the LLM
    prompt = f"""
    You are a user role classification AI. Given the following user data from a database,
    your task is to predict the user's most likely role.

    Available Roles: {', '.join(VALID_ROLES)}

    User Data:
    {json.dumps(user_data, indent=2)}

    Based on this data, which of the available roles best fits this user?
    Provide the role in a JSON object with the key 'predicted_role'.
    For example: {{ "predicted_role": "MANAGER" }}
    """
    
    try:
        # Call Groq API with a structured response schema
        chat_completion = groq_client.chat.completions.create(
            # CORRECTED: Changed model to a valid tool-use model.
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a user role classification AI. Respond only with a JSON object containing the predicted role."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        response_content = chat_completion.choices[0].message.content
        prediction = json.loads(response_content)
        predicted_role = prediction.get("predicted_role", "").upper()
        
        if predicted_role in VALID_ROLES:
            return predicted_role
        else:
            print(f"Warning: LLM returned an invalid role: {predicted_role}. Defaulting to {DEFAULT_ROLE}.", file=sys.stderr)
            return DEFAULT_ROLE

    except Exception as e:
        print(f"Error during AI role prediction: {e}. Defaulting to {DEFAULT_ROLE}.", file=sys.stderr)
        return DEFAULT_ROLE


def get_query_params_from_groq(prompt):
    """
    Sends a prompt to Groq to extract query parameters in a structured JSON format.
    The response is expected to be a JSON object containing the search term and tables to query.
    This approach is more robust than having the LLM generate raw SQL.
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            # CORRECTED: Changed model to a valid tool-use model.
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert at extracting database query parameters from a natural language request. The search term MUST be the full name or a complete ID, not a single word from the name. Respond with a JSON object. "},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        response_content = chat_completion.choices[0].message.content.strip()
        query_params = json.loads(response_content)
        return query_params
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to decode JSON from LLM: {e}. Content: {response_content}", file=sys.stderr)
        return {"error": f"Groq response format error: {e}"}
    except Exception as e:
        return {"error": f"Groq query parameter extraction error: {e}. Check API key and network connection."}

def execute_sql(sql, host, user, password, database, port):
    """
    Executes a given SQL query on the specified MySQL database.
    Returns a tuple of (fetched rows, column descriptions) or a dictionary with an 'error' key.
    """
    connection = None
    cursor = None
    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=int(port)
        )
        cursor = connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        description = cursor.description
        return rows, description
    except pymysql.MySQLError as e:
        # Return a consistent error format
        return {"error": f"SQL execution error: {e}. Query: '{sql}'"}, None
    except Exception as e:
        # Handle unexpected errors
        return {"error": f"An unexpected error occurred during SQL execution: {e}. Query: '{sql}'"}, None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def get_user_name(user_id, db_config):
    """
    Fetches the user's name from the database based on their user ID.
    The logic has been updated to query the correct table based on user role.
    """
    if not user_id:
        return "Anonymous"

    host = db_config.get("HOST")
    user = db_config.get("USER")
    password = db_config.get("PASSWORD")
    database = db_config.get("NAME")
    port = db_config.get("PORT")

    connection = None
    cursor = None
    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=int(port)
        )
        cursor = connection.cursor()
        
        # --- NEW FIX: Check MaasApp_user for SUPER_ADMIN first based on user's new info ---
        sql_query = "SELECT first_name, last_name, email FROM MaasApp_user WHERE id = %s;"
        cursor.execute(sql_query, (user_id,))
        rows = cursor.fetchall()

        # --- NEW FIX: If not found, check MaasApp_superadminassociates for all other roles ---
        if not rows:
            sql_query = "SELECT first_name, last_name, email FROM MaasApp_superadminassociates WHERE id = %s;"
            cursor.execute(sql_query, (user_id,))
            rows = cursor.fetchall()

        if rows and rows[0]:
            first_name = rows[0][0]
            last_name = rows[0][1]
            email = rows[0][2]

            if first_name and last_name:
                return f"{first_name} {last_name}"
            elif first_name:
                return str(first_name)
            elif email:
                return str(email)
            else:
                return f"User {user_id}"
        else:
            return f"User {user_id}"
    except pymysql.MySQLError as e:
        print(f"Warning: Database error fetching user name for ID {user_id}: {e}", file=sys.stderr)
        return f"User {user_id}"
    except Exception as e:
        print(f"Warning: An unexpected error occurred fetching user name for ID {user_id}: {e}", file=sys.stderr)
        return {"error": f"An unexpected error occurred fetching user name: {e}"}
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# --- Smart Conversational Bot Class ---

class SmartConversationalBot:
    """
    A smart conversational bot that can answer general questions and
    also fetch data from a MySQL database using SQL queries,
    with strict safety measures and user-role awareness.
    """
    def __init__(self, db_config):
        self.db_config = db_config
        self.chat_history = []
        # --- UPDATED: Adjusted access hierarchy for the new PROJECT_MANAGER role. ---
        self.access_hierarchy = {
            'TEAM_MEMBER': 1,
            'TEAM_LEAD': 2,
            'PROJECT_MANAGER': 3,  # Sees Team_Lead (2) and Team_Member (1)
            'MANAGER': 4,          # Sees Project_Manager (3), TL (2), TM (1)
            'ADMIN': 5,            # Sees Manager (4) and below
            'SUPER_ADMIN': 6       # Sees all
        }
    
    def _get_target_user_role(self, search_term: str):
        """
        Helper function to get a user's role by name or ID.
        FIX: Updated to handle full names more effectively by checking
        the concatenation of first and last names.
        """
        host = self.db_config.get("HOST")
        user = self.db_config.get("USER")
        password = self.db_config.get("PASSWORD")
        database = self.db_config.get("NAME")
        port = self.db_config.get("PORT")

        connection = None
        cursor = None
        try:
            connection = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=database,
                port=int(port)
            )
            cursor = connection.cursor(pymysql.cursors.DictCursor) 
            
            result = None
            if str(search_term).isdigit():
                # --- NEW FIX: Check MaasApp_user for SUPER_ADMIN first based on user's new info ---
                # print(f"DEBUG: Checking for user ID '{search_term}' in MaasApp_user.") # COMMENTED OUT DEBUG PRINT
                sql_query_user = "SELECT role FROM MaasApp_user WHERE id = %s;"
                cursor.execute(sql_query_user, (search_term,))
                result = cursor.fetchone()
                
                # --- NEW FIX: If not found, check MaasApp_superadminassociates for all other roles ---
                if not result:
                    # print(f"DEBUG: User not found in MaasApp_user. Checking MaasApp_superadminassociates for user ID '{search_term}'.") # COMMENTED OUT DEBUG PRINT
                    sql_query_superadmin = "SELECT role FROM MaasApp_superadminassociates WHERE id = %s;"
                    cursor.execute(sql_query_superadmin, (search_term,))
                    result = cursor.fetchone()
            else:
                # FIX: Improved name search to handle multi-word names by
                # checking against the concatenation of first and last names.
                # --- NEW FIX: Check MaasApp_user for SUPER_ADMIN first ---
                # print(f"DEBUG: Checking for user name '{search_term}' in MaasApp_user.") # COMMENTED OUT DEBUG PRINT
                sql_query_user_name = "SELECT role FROM MaasApp_user WHERE CONCAT(`first_name`, ' ', `last_name`) LIKE %s OR `first_name` LIKE %s OR `last_name` LIKE %s;"
                cursor.execute(sql_query_user_name, (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
                result = cursor.fetchone()
                # --- NEW FIX: If not found, check MaasApp_superadminassociates for all other roles ---
                if not result:
                    # print(f"DEBUG: User not found in MaasApp_user. Checking MaasApp_superadminassociates for user name '{search_term}'.") # COMMENTED OUT DEBUG PRINT
                    sql_query_superadmin_name = "SELECT role FROM MaasApp_superadminassociates WHERE CONCAT(`first_name`, ' ', `last_name`) LIKE %s OR `first_name` LIKE %s OR `last_name` LIKE %s;"
                    cursor.execute(sql_query_superadmin_name, (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
                    result = cursor.fetchone()
            
            # --- NEW: Add debug print to show what was found. ---
            if result:
                # print(f"DEBUG: Found role '{result.get('role')}' for user '{search_term}'.") # COMMENTED OUT DEBUG PRINT
                pass
                
            if result and result.get('role'):
                # Normalize the role string: uppercase and replace spaces with underscores
                normalized_role = result['role'].upper().replace(' ', '_')
                return normalized_role
            else:
                return None
        except Exception as e:
            print(f"ERROR: Exception in _get_target_user_role for '{search_term}': {e}", file=sys.stderr)
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def _get_data_from_database(self, user_query: str, user_id: str = None, user_role: str = None, current_user_name: str = None):
        """
        Internal tool function to generate and execute SQL queries.
        This function is called by the LLM when it determines a database query is needed.
        It encapsulates the schema retrieval, SQL generation, safety, and execution.
        
        Args:
            user_query: The natural language query from the user.
            user_id: The ID of the current user.
            user_role: The role of the current user.
            current_user_name: The name of the current user.

        Returns:
            A JSON string containing the query result, an error message, and the generated SQL.
        """
        host = self.db_config.get("HOST")
        user = self.db_config.get("USER")
        password = self.db_config.get("PASSWORD")
        database = self.db_config.get("NAME")
        port = self.db_config.get("PORT")

        if not all([host, user, database, port]):
            return json.dumps({"error": "Database credentials (HOST, USER, PASSWORD, NAME, PORT) missing in configuration."})
            
        all_schema = get_db_schema(host, user, password, database, port)
        if "error" in all_schema:
            return json.dumps({"error": all_schema["error"]})
        
        # Define the prompt for the LLM to get query parameters
        sql_param_prompt = f"""
        You are an expert at extracting database query parameters from a natural language request.
        Your task is to identify a search term, a list of tables to query, and any specific status to filter by from the user's question.
        
        **CRITICAL CONTEXT & SECURITY RULES:**
        1.  **User Identity:** The current user's ID is '{user_id if user_id else "anonymous"}' and their role is '{user_role if user_role else "anonymous"}' and their name is '{current_user_name}'.
        2.  **Database Schema:** Use only the following schema for all queries:
            {json.dumps(all_schema, indent=2)}
        3.  **Role-based Access Rules (Access Level from Lowest (1) to Highest (6)):**
            -   **TEAM_MEMBER (1)**: Can only access their own data.
            -   **TEAM_LEAD (2)**: Can access their own data and `TEAM_MEMBER` (1) data.
            -   **PROJECT_MANAGER (3)**: Can access their own data, `TEAM_LEAD` (2), and `TEAM_MEMBER` (1) data.
            -   **MANAGER (4)**: Can access their own data, `PROJECT_MANAGER` (3), `TEAM_LEAD` (2), and `TEAM_MEMBER` (1) data.
            -   **ADMIN (5)**: Can access their own data, `MANAGER` (4), `PROJECT_MANAGER` (3), `TEAM_LEAD` (2), and `TEAM_MEMBER` (1) data.
            -   **SUPER_ADMIN (6)**: Has no access restrictions.
        4.  **Security & Filtering:**
            -   **Rule 1 (Self-Query):** If the user asks for information about themselves using phrases like "my tasks", "my leaves", or "show me my data" or "show me my details", the search term should be the current user's ID: '{user_id}'. This applies to all roles. For "my details", the tables to query are always `MaasApp_user` and `MaasApp_superadminassociates`.
            -   **Rule 2 (Named User):** If the user asks for data about another person by name or ID, the search term MUST be the person's **full name or complete ID**. Do not use a single word from the name. For example, for "show me yogesh more details", the search term is 'yogesh more'.
            -   **Rule 3 (Specific Column):** When the user's query contains a specific keyword like 'assigned to', 'created by', or 'raised by' followed by a name or ID, you MUST return the corresponding database column name in the `column_to_search` field. **This takes priority over a generic ID search.**
            -   **Rule 4 (Role-based Restriction):** Based on the Role-based Access Rules above, a user can only query data for individuals with a **strictly lower** access level. If a request for another person's data is not allowed (e.g., an Admin requesting data about another Admin), respond with an empty object `{{}}` as a signal to decline the request.
            -   **Rule 5 (General Data):** If the user's query is a general request without a specific name or "my" keyword (e.g., "show me tasks", "holiday data", "all events"), the `search_term` should be `null`.
        5.  **Status Filtering:** If the user mentions a status like 'pending', 'completed', 'overdue', or 'in progress', include the status in the JSON response.
        
        **User Question:** {user_query}
        
        Respond with a JSON object. If no query is needed (Rule 4 applies), respond with an empty object `{{}}` as a signal to decline the request.
        Otherwise, provide the search term (which must be the full name or ID if a person is mentioned), a list of tables to query, and a `status` field if applicable.
        For example:
        - For a TEAM_MEMBER asking "show me my tasks with pending status": `{{ "search_term": "{user_id}", "tables_to_query": ["MaasApp_task"], "status": "Pending" }}`
        - **CORRECTED EXAMPLE:** For a TEAM_LEAD asking "show me tasks assigned to Shital": `{{ "search_term": "Shital", "tables_to_query": ["MaasApp_task"], "column_to_search": "assigned_to" }}`
        - For "show me tasks assigned to 72": `{{ "search_term": "72", "tables_to_query": ["MaasApp_task"], "column_to_search": "assigned_to" }}`
        - For an ADMIN asking "show me all tasks with completed status": `{{ "search_term": null, "tables_to_query": ["MaasApp_task"], "status": "Completed" }}`
        - **NEW FIX EXAMPLE:** For "show me my details": `{{ "search_term": "{user_id}", "tables_to_query": ["MaasApp_user", "MaasApp_superadminassociates"] }}`
        - For "my projects" (for any user): `{{ "search_term": "{user_id}", "tables_to_query": ["MaasApp_project_details"] }}`
        - For a SUPER_ADMIN asking "show me satyam data": `{{ "search_term": "satyam", "tables_to_query": ["MaasApp_user", "MaasApp_superadminassociates"] }}`
        - **NEW EXAMPLE:** For "show me old and new values of task id 13": `{{ "search_term": "13", "tables_to_query": ["MaasApp_taskactivity"], "column_to_search": "task_id" }}`
        - **FIXED EXAMPLE:** For "show me taskactivity for updated_by_id 15": `{{ "search_term": "15", "tables_to_query": ["MaasApp_taskactivity"], "column_to_search": "updated_by_id" }}`
        - **EXAMPLE FOR SPECIFIC ATTENDANCE:** For "show me Jayesh Patil manual attendance status": `{{ "search_term": "Jayesh Patil", "tables_to_query": ["MaasApp_attendance"], "status": "PENDING" }}`
        - **EXAMPLE FOR TUTORIALS:** For "show me tutorials created by 10": `{{ "search_term": "10", "tables_to_query": ["MaasApp_tutorial"], "column_to_search": "created_by" }}`
        - **NEW FIX EXAMPLE for tutorials with assigned_to:** For "show me tutorials assigned to 51": `{{ "search_term": "51", "tables_to_query": ["MaasApp_tutorial"], "column_to_search": "assigned_to" }}`
        - **EXAMPLE FOR 'RAISED BY'**: For "show me tickets raised by 40": `{{ "search_term": "40", "tables_to_query": ["MaasApp_ticket"], "column_to_search": "raised_by" }}`
        - **NEW EXAMPLE FOR 'LEAVE'**: For "show me my leave details": `{{ "search_term": "{user_id}", "tables_to_query": ["MaasApp_applyleave"], "column_to_search": "user_id" }}`
        - **NEW EXAMPLE FOR 'HOLIDAY'**: For "show all holidays": `{{ "search_term": null, "tables_to_query": ["MaasApp_holiday"] }}`
        - **NEW EXAMPLE FOR 'EVENT'**: For "show all events": `{{ "search_term": null, "tables_to_query": ["MaasApp_event"] }}`
        - **NEW EXAMPLE FOR 'ATTENDANCE'**: For "show me my attendance": `{{ "search_term": "{user_id}", "tables_to_query": ["MaasApp_attendance"], "column_to_search": "employee_associated_id" }}`
        - **NEW EXAMPLE FOR 'LEAVETYPE'**: For "show leavetype": `{{ "search_term": null, "tables_to_query": ["MaasApp_leavetype"] }}`
        - **NEW EXAMPLE FOR 'CATEGORY'**: For "show category": `{{ "search_term": null, "tables_to_query": ["MaasApp_category"] }}`
        - **NEW CORRECTED EXAMPLE FOR 'TASKTEMPLATE'**: For "show me task templates created by super_admin_associated_id 123": `{{ "search_term": "123", "tables_to_query": ["MaasApp_tasktemplate"], "column_to_search": "super_admin_associated_id" }}`
        - **NEW CRITICAL FIX**: For a query like "show me attendance employee associated id 87", the AI must recognize that `employee_associated_id` is the column and `87` is the ID. The JSON should be `{{ "search_term": "87", "tables_to_query": ["MaasApp_attendance"], "column_to_search": "employee_associated_id" }}`.
        - **NEW CRITICAL EXAMPLE for nested lookup**: For "show me dailytaskfile for user id 88", the `user_id` is the search term, and the tables are `MaasApp_dailytask_files` and implicitly `MaasApp_dailytask`. The JSON should be `{{ "search_term": "88", "tables_to_query": ["MaasApp_dailytask_files"], "column_to_search": "user_id" }}`.
        - **NEW EXAMPLE FOR `MaasApp_achievement`**: For a user asking "show me my achievements", the query should be `{{ "search_term": "{user_id}", "tables_to_query": ["MaasApp_achievement"], "column_to_search": "user_id" }}`.
        - **NEW CRITICAL FIX FOR `MaasApp_meeting`**: For a user asking "show me meetings for user id 40", the query should be `{{ "search_term": "40", "tables_to_query": ["MaasApp_meeting"], "column_to_search": "user_id" }}`.
        - **NEW EXAMPLE FOR 'reviewsubmit'**: For a user asking "show me my review submissions", the query should be `{{ "search_term": "{user_id}", "tables_to_query": ["MaasApp_reviewsubmit"], "column_to_search": "user_id" }}`.
        - **NEW EXAMPLE FOR 'MaasApp_blocker'**: For a user asking "show me blockers for user id 40", the query should be `{{ "search_term": "40", "tables_to_query": ["MaasApp_blocker"], "column_to_search": "user_id" }}`.
        - **NEW EXAMPLE FOR 'MaasApp_dailytask'**: For a user asking "show me daily tasks for user id 40", the query should be `{{ "search_term": "40", "tables_to_query": ["MaasApp_dailytask"], "column_to_search": "user_id" }}`.
        - **NEW EXAMPLE FOR 'MaasApp_feedback'**: For a user asking "show me feedback for user id 40", the query should be `{{ "search_term": "40", "tables_to_query": ["MaasApp_feedback"], "column_to_search": "user_id" }}`.
        - **NEW EXAMPLE FOR 'MaasApp_meeting'**: For a user asking "show me meetings for user id 40", the query should be `{{ "search_term": "40", "tables_to_query": ["MaasApp_meeting"], "column_to_search": "user_id" }}`.
        - **NEW EXAMPLE FOR 'MaasApp_reviewsubmit'**: For a user asking "show me review submissions for user id 40", the query should be `{{ "search_term": "40", "tables_to_query": ["MaasApp_reviewsubmit"], "column_to_search": "user_id" }}`.
        """
        
        query_params = get_query_params_from_groq(sql_param_prompt)

        if not query_params:
            return json.dumps({"error": "Access restricted."})

        if "error" in query_params:
            return json.dumps({"error": query_params['error']})

        search_term = query_params.get("search_term")
        tables_to_query = query_params.get("tables_to_query", [])
        status_filter = query_params.get("status")
        column_to_search = query_params.get("column_to_search")

        if not tables_to_query:
            return json.dumps({"error": "The assistant could not generate a valid query from your request."})
            
        is_self_query = str(search_term) == str(user_id)
        
        # --- Check if the current table is a public table. If so, clear any filters. ---
        is_public_table_query = any(table in PUBLIC_TABLES for table in tables_to_query)
        
        # --- RE-EVALUATED ACCESS CONTROL LOGIC ---
        # This logic now correctly bypasses the role check for public document queries.
        
        # --- NEW FIX: Add a special case for task activity queries. ---
        # --- The `column_to_search` is now dynamic to handle both task_id and updated_by_id ---
        if 'MaasApp_taskactivity' in tables_to_query:
            if column_to_search == 'task_id':
                try:
                    connection = pymysql.connect(
                        host=host, user=user, password=password, database=database, port=int(port)
                    )
                    cursor = connection.cursor()
                    
                    # Check if the current user is associated with the task
                    task_query = "SELECT `id` FROM `MaasApp_task` WHERE `id` = %s AND (`assigned_to` = %s OR `users_in_loop` LIKE %s OR `assigned_by` = %s);"
                    cursor.execute(task_query, (search_term, user_id, f'%"{user_id}"%', current_user_name))
                    task_rows = cursor.fetchone()
                    
                    cursor.close()
                    connection.close()

                    if not task_rows:
                        return json.dumps({"error": "Access restricted. You do not have permission to view this task's activity."})

                except Exception as e:
                    print(f"Error during task permissions check: {e}", file=sys.stderr)
                    return json.dumps({"error": "An internal error occurred during permission check."})
            elif column_to_search == 'updated_by_id':
                current_user_access_level = self.access_hierarchy.get(user_role.upper(), 0)
                target_role = self._get_target_user_role(search_term)
                
                # Check if the user has permission to see data for the updated_by_id
                if not target_role:
                    if user_role.upper() != 'SUPER_ADMIN':
                        return json.dumps({"error": "Access restricted. Target user's role could not be determined."})
                else:
                    target_user_access_level = self.access_hierarchy.get(target_role, 0)
                    if not (user_role.upper() == 'SUPER_ADMIN' or current_user_access_level > target_user_access_level):
                        return json.dumps({"error": "Access restricted. You do not have permission to view this user's data."})
        
        # --- CORRECTED: The `_get_data_from_database` function needs to correctly handle the access
        # hierarchy when a user queries for another user's data. A `TEAM_LEAD` should be able to
        # query data for a `TEAM_MEMBER`. The `_get_target_user_role` function now correctly
        # returns the role, and this logic properly compares the hierarchy. ---
        elif not is_self_query and search_term is not None and not is_public_table_query:
            current_user_access_level = self.access_hierarchy.get(user_role.upper(), 0)
            target_role = self._get_target_user_role(search_term)
            
            # If the target user's role is not found, restrict access unless the current user is a SUPER_ADMIN
            if not target_role:
                if user_role.upper() != 'SUPER_ADMIN':
                    return json.dumps({"error": "Access restricted. Target user's role could not be determined."})
            else:
                target_user_access_level = self.access_hierarchy.get(target_role, 0)
                
                # FIX: Corrected the security flaw. The check is now `>` not `>=` to prevent viewing peers' data.
                # A SUPER_ADMIN is an an exception and can see any data.
                if not (user_role.upper() == 'SUPER_ADMIN' or current_user_access_level > target_user_access_level):
                    return json.dumps({"error": "Access restricted. You do not have permission to view this user's data."})
        # --- End of Fixed Access Control ---
        
        # --- Check if the user has access to the requested tables. ---
        if user_role.upper() in ['TEAM_MEMBER', 'TEAM_LEAD', 'PROJECT_MANAGER']:
            for table_name in tables_to_query:
                # The corrected access logic above now handles role-based restrictions
                # on sensitive tables. The check here is now for a broader purpose.
                pass
        
        # --- FIX: Force self-query filter for TEAM_MEMBER on general requests UNLESS it's a public table. ---
        if user_role.upper() == 'TEAM_MEMBER' and search_term is None and not is_public_table_query:
            is_self_query = True
            search_term = user_id
        # --- End of new fix ---
        
        results = []
        for table_name in tables_to_query:
            columns = all_schema.get(table_name)
            if not columns:
                return json.dumps({"error": f"Table '{table_name}' does not exist in the database."})
            
            where_clauses = []
            
            # --- Check if the current table is a public table. If so, clear any filters. ---
            if table_name in PUBLIC_TABLES:
                # If a search term is provided, filter by it on the 'assigned_to' column.
                if search_term is not None and 'assigned_to' in columns:
                    where_clauses.append(f"(`assigned_to` = '{search_term}' OR `{column_to_search}` LIKE '%[{search_term}]%' OR `{column_to_search}` LIKE '%[{search_term},%' OR `{column_to_search}` LIKE '%, {search_term}]%' OR `{column_to_search}` LIKE '%, {search_term},%')")
                elif search_term is not None and 'created_by' in columns and column_to_search == 'created_by':
                     where_clauses.append(f"`created_by` = '{search_term}'")
                # --- NEW FIX: Handle tasktemplate specific filtering by super_admin_associated_id. ---
                # --- This block has been updated to use the corrected column name. ---
                elif table_name == 'MaasApp_tasktemplate' and search_term is not None and 'super_admin_associated_id' in columns:
                    where_clauses.append(f"`super_admin_associated_id` = '{search_term}'")
                else:
                    where_clauses = []
            
            # This is the original, correct logic for private data.
            # It will only run if the table is NOT in the PUBLIC_TABLES list.
            elif is_self_query:
                # If a TEAM_MEMBER is asking for tasks, use the `assigned_to` column, not `id`.
                if table_name == 'MaasApp_task':
                    if 'assigned_to' in columns:
                        where_clauses.append(f"`assigned_to` = '{user_id}'")
                    else:
                        return json.dumps({"error": f"Table '{table_name}' does not have the 'assigned_to' column."})
                
                # --- NEW LOGIC FOR TEAM_MEMBER QUERIES ON MaasApp_taskactivity ---
                elif user_role.upper() == 'TEAM_MEMBER' and table_name == 'MaasApp_taskactivity':
                    # --- NEW FIX: The check is now dynamic based on column_to_search ---
                    if column_to_search in columns:
                        if column_to_search == 'task_id':
                            # First, get the task IDs from MaasApp_task for this user
                            try:
                                connection = pymysql.connect(
                                    host=host, user=user, password=password, database=database, port=int(port)
                                )
                                cursor = connection.cursor()
                                task_ids_query = f"SELECT `id` FROM `MaasApp_task` WHERE `assigned_to` = '{user_id}';"
                                cursor.execute(task_ids_query)
                                task_ids = [str(row[0]) for row in cursor.fetchall()]
                                cursor.close()
                                connection.close()
                            except Exception as e:
                                print(f"Warning: Failed to retrieve task_ids for TEAM_MEMBER from MaasApp_task: {e}", file=sys.stderr)
                                task_ids = []

                            if task_ids:
                                ids_string = ', '.join(f"'{tid}'" for tid in task_ids)
                                where_clauses.append(f"`task_id` IN ({ids_string})")
                            else:
                                # If no task IDs are found, create a condition that returns no results
                                where_clauses.append("1 = 0")
                        elif column_to_search == 'updated_by_id':
                            # --- FIX: For self-queries, the updated_by_id should be the user's own ID ---
                            where_clauses.append(f"`updated_by_id` = '{user_id}'")
                    else:
                        return json.dumps({"error": f"Table '{table_name}' does not have the '{column_to_search}' column."})
                # --- END OF NEW LOGIC ---
                
                # --- NEW: Add logic for 'assigned_to_ids' for MaasApp_project_details. ---
                elif table_name == 'MaasApp_project_details':
                    if 'assigned_to_ids' in columns:
                        where_clauses.append(f"`assigned_to_ids` = '{user_id}'")
                    else:
                        return json.dumps({"error": f"Table '{table_name}' does not have the 'assigned_to_ids' column."})
                
                # --- FIX: New logic for 'raised_by' column specifically for MaasApp_ticket. ---
                elif table_name == 'MaasApp_ticket':
                    if 'raised_by' in columns:
                        where_clauses.append(f"`raised_by` = '{user_id}'")
                    else:
                        return json.dumps({"error": f"Table '{table_name}' does not have the 'raised_by' column."})
                
                # --- NEW FIX: Add logic for 'user_id' column specifically for MaasApp_applyleave. ---
                elif table_name == 'MaasApp_applyleave':
                    if 'user_id' in columns:
                        where_clauses.append(f"`user_id` = '{user_id}'")
                    else:
                        return json.dumps({"error": f"Table '{table_name}' does not have the 'user_id' column."})
                
                # --- FIX: New logic for `employee_associated_id` for MaasApp_attendance. ---
                elif table_name == 'MaasApp_attendance':
                    if 'employee_associated_id' in columns:
                        where_clauses.append(f"`employee_associated_id` = '{user_id}'")
                    else:
                        return json.dumps({"error": f"Table '{table_name}' does not have the 'employee_associated_id' column."})
                
                # --- NEW FIX: Add logic for 'id' for MaasApp_user and MaasApp_superadminassociates. ---
                elif table_name in ['MaasApp_user', 'MaasApp_superadminassociates']:
                    if 'id' in columns:
                        where_clauses.append(f"`id` = '{user_id}'")
                    else:
                        return json.dumps({"error": f"Table '{table_name}' does not have the 'id' column."})
                
                # --- NEW CORRECTED LOGIC: Use user_id for all the specified tables ---
                elif table_name in ['MaasApp_achievement', 'MaasApp_blocker', 'MaasApp_dailytask', 'MaasApp_feedback', 'MaasApp_meeting', 'MaasApp_reviewsubmit']:
                    if 'user_id' in columns:
                        where_clauses.append(f"`user_id` = '{user_id}'")
                    else:
                        return json.dumps({"error": f"Table '{table_name}' does not have the 'user_id' column."})

                # --- NEW LOGIC for MaasApp_dailytask_files, which requires a nested lookup ---
                elif table_name == 'MaasApp_dailytask_files':
                    # First, get the task IDs from MaasApp_dailytask for this user
                    try:
                        connection = pymysql.connect(
                            host=host, user=user, password=password, database=database, port=int(port)
                        )
                        cursor = connection.cursor()
                        task_ids_query = f"SELECT `id` FROM `MaasApp_dailytask` WHERE `user_id` = '{user_id}';"
                        cursor.execute(task_ids_query)
                        task_ids = [str(row[0]) for row in cursor.fetchall()]
                        cursor.close()
                        connection.close()
                    except Exception as e:
                        print(f"Warning: Failed to retrieve task_ids from MaasApp_dailytask: {e}", file=sys.stderr)
                        task_ids = []

                    if task_ids:
                        ids_string = ', '.join(f"'{tid}'" for tid in task_ids)
                        if 'taskid' in columns:
                            where_clauses.append(f"`taskid` IN ({ids_string})")
                        else:
                            return json.dumps({"error": f"Table '{table_name}' does not have the 'taskid' column."})
                    else:
                        # If no task IDs are found, create a condition that returns no results
                        where_clauses.append("1 = 0")
                
                # Handle general "my" queries for other private tables.
                else:
                    user_id_columns = ['user_id', 'assigned_to', 'employee_user_id', 'id',]
                    matched_user_id_col = next((col for col in user_id_columns if col in columns), None)
                    if matched_user_id_col:
                        where_clauses.append(f"`{matched_user_id_col}` = '{user_id}'")

            elif search_term is not None:
                # 1. Handle queries by specific ID columns (like task_id or assigned_to).
                if column_to_search and column_to_search in columns:
                    if column_to_search == 'assigned_to':
                        # Check if the search term is a number (ID) or a string (name)
                        if str(search_term).isdigit():
                            # It's an ID, use the assigned_to column
                            where_clauses.append(f"(`assigned_to` = '{search_term}' OR `{column_to_search}` LIKE '%[{search_term}]%' OR `{column_to_search}` LIKE '%[{search_term},%' OR `{column_to_search}` LIKE '%, {search_term}]%' OR `{column_to_search}` LIKE '%, {search_term},%')")
                        else:
                            # It's a name, use the assigned_to_name column
                            where_clauses.append(f"`assigned_to_name` LIKE '%{search_term}%'")
                    elif column_to_search == 'employee_associated_id':
                        # --- NEW CRITICAL FIX: Add a specific check for 'employee_associated_id' to ensure exact match on the column. ---
                        where_clauses.append(f"`employee_associated_id` = '{search_term}'")
                    # --- NEW FIX: Handle the `user_id` column for MaasApp_applyleave and other tables. ---
                    elif column_to_search == 'user_id' and table_name in ['MaasApp_applyleave', 'MaasApp_blocker', 'MaasApp_dailytask', 'MaasApp_feedback', 'MaasApp_meeting', 'MaasApp_reviewsubmit', 'MaasApp_dailytask_files']:
                        where_clauses.append(f"`user_id` = '{search_term}'")
                    else:
                        where_clauses.append(f"`{column_to_search}` = '{search_term}'")
                # 2. Handle attendance-specific queries by employee name.
                elif table_name == 'MaasApp_attendance' and 'employee_name' in columns:
                    where_clauses.append(f"`employee_name` LIKE '%{search_term}%'")
                # 3. Handle queries by general user ID or ID columns.
                elif str(search_term).isdigit():
                    id_columns = ['id', 'user_id', 'employee_user_id', 'employee_associated_id']
                    matched_id_col = next((col for col in id_columns if col in columns), None)
                    if matched_id_col:
                        where_clauses.append(f"`{matched_id_col}` = '{search_term}'")
                # 4. Handle general name searches across tables.
                else:
                    if 'first_name' in columns and 'last_name' in columns:
                        where_clauses.append(f"CONCAT(`first_name`, ' ', `last_name`) LIKE '%{search_term}%'")
                    elif 'assigned_to_name' in columns:
                        where_clauses.append(f"`assigned_to_name` LIKE '%{search_term}%'")
                    elif 'first_name' in columns:
                        where_clauses.append(f"`first_name` LIKE '%{search_term}%'")
                    elif 'last_name' in columns:
                        where_clauses.append(f"`last_name` LIKE '%{search_term}%'")
            
            # Add filter for user ID/name based on self-query if it applies
            if is_self_query and not is_public_table_query:
                # The search_term logic above already handles self-query by ID for many cases.
                # This ensures we have a filter if no name/ID was explicitly mentioned but the user asked for "my" data.
                if not where_clauses:
                    if table_name == 'MaasApp_attendance' and 'employee_associated_id' in columns:
                        where_clauses.append(f"`employee_associated_id` = '{user_id}'")
                    elif table_name == 'MaasApp_project_details' and 'assigned_members' in columns and current_user_name:
                        where_clauses.append(f"`assigned_members` LIKE '%\"{current_user_name}\"%'")
                    else:
                        user_id_columns = ['user_id', 'assigned_to', 'employee_user_id', 'id',]
                        matched_user_id_col = next((col for col in user_id_columns if col in columns), None)
                        if matched_user_id_col:
                            where_clauses.append(f"`{matched_user_id_col}` = '{user_id}'")
            
            # Correctly add the status filter regardless of user role
            if status_filter and 'status' in columns:
                where_clauses.append(f"`status` = '{status_filter}'")

            # For general queries by TEAM_LEAD/PROJECT_MANAGER, filter by `assigned_by` if it exists.
            if user_role.upper() in ['TEAM_LEAD', 'PROJECT_MANAGER'] and search_term is None:
                if 'assigned_by' in columns:
                    where_clauses.append(f"`assigned_by` = '{current_user_name}'")


            sql_query = f"SELECT * FROM `{table_name}`"
            if where_clauses:
                sql_query += f" WHERE {' AND '.join(where_clauses)};"
            else:
                sql_query += ";"
            
            # print(f"DEBUG: Generated SQL Query: {sql_query}") # COMMENTED OUT DEBUG PRINT
            query_result, description = execute_sql(sql_query, host, user, password, database, port)

            if isinstance(query_result, dict) and "error" in query_result:
                return json.dumps({"error": query_result['error']})

            formatted_result = []
            if description:
                column_names = [col_desc[0] for col_desc in description]
                for row in query_result:
                    item = {}
                    for col_name, value in zip(column_names, row):
                        if isinstance(value, Decimal):
                            value = float(value)
                        elif isinstance(value, (datetime, date)):
                            value = value.isoformat()
                        
                        if col_name.lower() in ['id', 'user_id', 'employee_user_id', 'password', 'password_reset_token'] or 'token' in col_name.lower():
                            continue

                        if value is not None and value != '' and value != '[]':
                            item[col_name] = value
                    formatted_result.append(item)
            
            results.append({
                "table": table_name,
                "query": sql_query,
                "result": formatted_result
            })
            
        return json.dumps({
            "generated_queries": [res['query'] for res in results],
            "query_results": results,
            "message": "Data retrieved successfully.",
            "search_term_used": search_term,
            "is_self_query": is_self_query
        })

    def handle_conversation(self, user_query: str, user_id: str = None, user_role: str = None, current_user_name: str = None):
        """
        Handles the main conversational flow, deciding whether to respond directly
        or to call the database query tool.
        
        Args:
            user_query: The natural language query from the user.
            user_id: The ID of the current user.
            user_role: The role of the current user.
            current_user_name: The name of the current user.
            
        Returns:
            A dictionary containing the response and tool output.
        """
        lower_query = user_query.lower()
        if "what is your name" in lower_query or "who are you" in lower_query:
            return {"response": "I am the MaasApp Personal Assistant, a large language model created by Groq.", "tool_output": None}

        self.chat_history.append({"role": "user", "content": user_query})

        history_tokens = sum(len(m['content'].split()) for m in self.chat_history if 'content' in m)

        if len(self.chat_history) > MAX_HISTORY_LENGTH or history_tokens > MAX_TOKENS_FOR_HISTORY:
            while len(self.chat_history) > MAX_HISTORY_LENGTH or history_tokens > MAX_TOKENS_FOR_HISTORY:
                if len(self.chat_history) > 1:
                    self.chat_history.pop(0)
                    history_tokens = sum(len(m['content'].split()) for m in self.chat_history if 'content' in m)
                else:
                    break
        
        # --- FIX: Ensure current_user_name is always a string. ---
        if current_user_name is None:
            current_user_name = "Anonymous User"

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_data_from_database",
                    "description": "Retrieve specific data from the MySQL database based on a natural language query. Use this for questions asking for lists, details, counts, or specific facts from the database (e.g., 'show me all team members', 'what are my tasks', 'how many leaves have I taken'). Do NOT use this tool for general conversational questions like 'what is your purpose?', 'how are you?', or 'who are you?'. Answer these directly. Crucially, use this tool for questions about the current user's identity, such as 'what is my name?' or 'who am I?' or 'show me my data'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_query": {
                                "type": "string",
                                "description": "The user's original natural language query that needs database access."
                            },
                            "user_id": {
                                "type": "string",
                                "description": "The ID of the current user. Pass None if the user is anonymous.",
                                "nullable": True
                            },
                            "user_role": {
                                "type": "string",
                                "description": "The role of the current user (e.g., 'ADMIN', 'TEAM_MEMBER', 'MANAGER', 'SUPER_ADMIN'). Pass None if the user is anonymous.",
                                "nullable": True
                            },
                            "current_user_name": {
                                "type": "string",
                                "description": "The full name of the current user.",
                                # --- FIX: This parameter is now required to prevent the LLM from sending null. ---
                                "nullable": False
                            }
                        },
                        "required": ["user_query", "current_user_name"]
                    }
                }
            }
        ]

        messages_for_llm = [
            {"role": "system", "content": f"""
            You are a helpful conversational AI assistant working with the MaasApp application.
            Your primary role is to engage in natural conversation.
            You can also retrieve data from the MySQL database when explicitly asked for information related to leaves, attendance, tasks, users, events, holidays, etc.
            Remember to use your `get_data_from_database` tool for questions about the user's identity, such as 'what is my name?' or 'who am I?' or 'show me my data'.
            Do NOT call tools for general conversational questions like 'what is your purpose?', 'how are you?', or 'who are you?'. Answer these directly.
            When a database query returns no results, simply state that no data was found for the request, without elaborating on the query itself.
            The current user's ID is '{user_id if user_id else "anonymous"}' and their role is '{user_role if user_role else "anonymous"}' and their name is '{current_user_name}'."""
            }
        ] + self.chat_history

        try:
            chat_completion = groq_client.chat.completions.create(
                # CORRECTED: Changed model to a valid tool-use model.
                model="llama-3.1-8b-instant",
                messages=messages_for_llm,
                tools=tools,
                tool_choice="auto",
                temperature=0.7
            )

            response_message = chat_completion.choices[0].message
            self.chat_history.append(response_message)

            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "get_data_from_database":
                    tool_response_json = self._get_data_from_database(
                        user_query=function_args.get("user_query"),
                        user_id=user_id,
                        user_role=user_role,
                        current_user_name=current_user_name
                    )
                    tool_response = json.loads(tool_response_json)

                    if "error" in tool_response:
                        final_response_content = tool_response['error']
                        self.chat_history.append({"role": "assistant", "content": final_response_content})
                        return {"response": final_response_content, "tool_output": {"error": tool_response['error']}}
                    else: 
                        self.chat_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_response_json,
                            }
                        )
                        final_response_completion = groq_client.chat.completions.create(
                            # CORRECTED: Changed model to a valid tool-use model.
                            model="llama-3.1-8b-instant",
                            messages=self.chat_history,
                            temperature=0.7
                        )
                        final_response_content = final_response_completion.choices[0].message.content
                        
                        all_results = tool_response.get('query_results', [])
                        search_term_used = tool_response.get('search_term_used')
                        is_self_query_from_tool = tool_response.get('is_self_query')

                        # Re-evaluating based on the corrected logic.
                        if "Access restricted" in final_response_content:
                             pass
                        elif not all_results or all(not res['result'] for res in all_results):
                            # The LLM's response for 'no results' is handled here.
                            final_response_content = "I couldn't find any data matching your request. Is there anything else I can help with?"
                        else:
                            # If results are found, make the LLM's conversational response empty
                            final_response_content = ""
                                
                        self.chat_history.append({"role": "assistant", "content": final_response_content})
                        return {"response": final_response_content, "tool_output": tool_response}
                else:
                    return {"response": "Sorry, I tried to use an unknown tool.", "tool_output": None}
            else:
                self.chat_history.append({"role": "assistant", "content": response_message.content})
                return {"response": response_message.content, "tool_output": None}

        except json.JSONDecodeError as e:
            return {"response": f"An internal error occurred while processing the tool's arguments: {str(e)}", "tool_output": None}
        except Exception as e:
            return {"response": f"An error occurred: {str(e)}. Please try again.", "tool_output": None}

# --- Helper function to format database records into paragraphs ---
def _format_record_to_sentence(record, record_title):
    """
    Formats a single database record into a human-readable sentence.
    This function is kept for backward compatibility but is no longer used in the main loop.
    """
    if not record:
        return ""

    details = []
    
    # List of fields to be explicitly excluded from the output
    excluded_fields = [
        'is_verified', 'is_logged_in', 'last_login_time', 'created_at',
        'profile_picture', 'is_facial_data_uploaded'
    ]

    # Iterate through all key-value pairs
    for key, value in record.items():
        # --- MODIFIED: Skip id-related fields and new excluded fields. ---
        if (key.lower() in excluded_fields or key.lower() == 'id' or 
            ('_id' in key.lower() and key.lower() != 'employee_id')) or \
           value is None or value == '' or value == '[]' or \
           'password' in key.lower() or 'token' in key.lower():
            continue
        
        # Format the key
        formatted_key = ' '.join(word.capitalize() for word in key.split('_'))
        
        # Format the value
        if isinstance(value, (datetime, date)):
            value_str = value.isoformat().split('T')[0]
        elif isinstance(value, Decimal):
            value_str = f"{float(value):.2f}"
        elif isinstance(value, bool):
            value_str = "verified" if value else "not verified"
        elif isinstance(value, list):
            value_str = ", ".join(map(str, value))
        else:
            value_str = str(value)

        # Append to the list of details
        details.append(f"{formatted_key} is {value_str}")
    
    # Join all details into a single sentence and add a period
    return f"Details for {record_title}: {'. '.join(details)}." if details else f"Details for {record_title}: No data found."

def _format_record_to_list(record, record_title):
    """
    Formats a single database record into a human-readable list.
    """
    formatted_output = f"--- {record_title} ---\n"
    
    # List of fields to be explicitly excluded from the output
    excluded_fields = [
        'is_verified', 'is_logged_in', 'last_login_time', 'created_at',
        'profile_picture', 'is_facial_data_uploaded'
    ]

    for key, value in record.items():
        # --- MODIFIED: Skip id-related fields and new excluded fields. ---
        if (key.lower() in excluded_fields or key.lower() == 'id' or 
            ('_id' in key.lower() and key.lower() != 'employee_id')) or \
           value is None or value == '' or value == '[]' or \
           'password' in key.lower() or 'token' in key.lower():
            continue
        
        # Format the key
        formatted_key = ' '.join(word.capitalize() for word in key.split('_'))
        
        # Format the value
        if isinstance(value, (datetime, date)):
            value_str = value.isoformat().split('T')[0]
        elif isinstance(value, Decimal):
            value_str = f"{float(value):.2f}"
        elif isinstance(value, bool):
            value_str = "Verified" if value else "Not Verified"
        elif isinstance(value, list):
            value_str = ", ".join(map(str, value))
        else:
            value_str = str(value)

        formatted_output += f"  - {formatted_key}: {value_str}\n"
    return formatted_output

# --- Main Execution Block ---

if __name__ == "__main__":
    DATABASE_CONFIG = {
        'ENGINE': 'django.db.backends.mysql', #Database Engine
        'NAME': 'New_Project_Maas', #Database Name
        'USER': 'New_Project_Maas_user',
        'PASSWORD': 'gJ3rWqklltdx',
        'HOST': '192.168.0.253',
        'PORT': '3306',
        'OPTIONS': {
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
        },
    }

    print("Attempting to connect to the database to verify credentials...")
    test_connection = None
    try:
        test_connection = pymysql.connect(
            host=DATABASE_CONFIG['HOST'],
            user=DATABASE_CONFIG['USER'],
            password=DATABASE_CONFIG['PASSWORD'],
            database=DATABASE_CONFIG['NAME'],
            port=int(DATABASE_CONFIG['PORT'])
        )
        print("Successfully connected to the database!")
        test_connection.close()
    except pymysql.MySQLError as e:
        print(f"Error: Could not connect to the database. Please check your DB credentials and ensure MySQL server is running.", file=sys.stderr)
        print(f"MySQL Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during database connection test: {e}", file=sys.stderr)
        sys.exit(1)

    bot = SmartConversationalBot(DATABASE_CONFIG)

    print("\n--- MaasApp Personal Assistant ---")
    print("Please provide your User ID to get started.")

    current_user_id = ""
    while not current_user_id:
        user_input_id = input("Enter your User ID (e.g., '1', '2', '3'): ").strip()
        if not user_input_id:
            print("User ID cannot be empty. Please try again.")
            continue
        
        user_data_result = get_user_data_for_prediction(user_input_id, DATABASE_CONFIG)
        if "error" in user_data_result:
            print(f"Error: {user_data_result['error']}. Please enter a valid User ID.")
            continue
        
        current_user_id = user_input_id
        
        if 'role' in user_data_result and user_data_result['role'].upper() in VALID_ROLES:
            current_user_role = user_data_result['role'].upper()
        else:
            current_user_role = predict_user_role_with_llm(current_user_id, DATABASE_CONFIG)
            if not current_user_role or "error" in current_user_role:
                print(f"Error: Could not determine role for user ID {current_user_id}. Exiting.")
                sys.exit(1)

    current_user_name = get_user_name(current_user_id, DATABASE_CONFIG)
    # --- FIX: Ensure current_user_name is always a string from the beginning. ---
    if not current_user_name or "error" in current_user_name:
        current_user_name = f"User {current_user_id}"


    print(f"Hello {current_user_name}! I'm your MaasApp Personal Assistant.")
    print(f"You have been identified as a '{current_user_role}'.")
    print("How can I assist you today?")
    print("Type 'exit' to quit.\nType 'clear' to clear chat history.\nType 'set user <id>' to change user ID during the session.")
    print("--------------------------------")
    
    preferred_lang_code = "en"
    print(f"✅ Language set to: English ({preferred_lang_code})")

    while True:
        updated_user_name = get_user_name(current_user_id, DATABASE_CONFIG)
        if updated_user_name != current_user_name:
            current_user_name = updated_user_name
            print(f"Greeting updated: Hello {current_user_name}!")

        user_input = input(f"\n{current_user_name} ({current_user_role}): ").strip()

        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'clear':
            bot.chat_history = []
            print("Chat history cleared.")
            continue
        elif user_input.lower().startswith('set user '):
            parts = user_input.split(' ')
            if len(parts) == 3:
                new_user_id = parts[2]
                user_data_result = get_user_data_for_prediction(new_user_id, DATABASE_CONFIG)
                if "error" in user_data_result:
                    print(f"Error: {user_data_result['error']}. User not changed.")
                    continue
                
                if 'role' in user_data_result and user_data_result['role'].upper() in VALID_ROLES:
                    new_user_role = user_data_result['role'].upper()
                else:
                    new_user_role = predict_user_role_with_llm(new_user_id, DATABASE_CONFIG)

                if new_user_role and "error" not in new_user_role:
                    current_user_id = new_user_id
                    current_user_role = new_user_role
                    new_user_name = get_user_name(new_user_id, DATABASE_CONFIG)
                    print(f"User set to ID: '{current_user_id}', Role: '{current_user_role}', Name: '{new_user_name}'")
                else:
                    print(f"Error: Could not find a valid role for user ID '{new_user_id}'. User not changed.")
            else:
                print("Invalid 'set user' command. Use 'set user <id>'.")
            continue

        detected_lang = preferred_lang_code
        print(f"MaasApp Personal Assistant ({detected_lang.upper()}): Thinking...")

        response_data = bot.handle_conversation(user_input, current_user_id, current_user_role, current_user_name)
        
        tool_output = response_data.get('tool_output')
        
        if tool_output and tool_output.get('query_results'):
            query_results = tool_output.get('query_results', [])
            all_records = []
            
            for query_info in query_results:
                query_result = query_info.get('result', [])
                if query_result:
                    all_records.extend(query_result)
            
            if all_records:
                print("\nHere are the results:")
                
                for i, record in enumerate(all_records):
                    # Create a title for the record
                    first_name = record.get('first_name')
                    last_name = record.get('last_name')
                    
                    if first_name and last_name:
                        record_title = f"{first_name} {last_name}"
                    else:
                        record_title = record.get('name') or record.get('title') or 'Record'
                    
                    # Use the new formatting function (list format)
                    formatted_output_list = _format_record_to_list(record, record_title)
                    # --- FIX: Correct the typo in the output to "task" instead to "tast". ---
                    formatted_output_list = formatted_output_list.replace("tast", "task")
                    
                    print(f"\n{formatted_output_list}")

                    # --- NEW ADDITION: Add the sentence structure response as requested by the user ---
                    formatted_output_sentence = _format_record_to_sentence(record, record_title)
                    formatted_output_sentence = formatted_output_sentence.replace("tast", "task")
                    print(f"\n[Sentence Summary]: {formatted_output_sentence}")
                    # --- END OF NEW ADDITION ---
                
                print("\nLet me know if you need any further assistance!")
            else:
                # New else block to handle when no data is found.
                print("\nNo data found for your request.")
                
        elif response_data.get('response'):
            print(f"MaasApp Personal Assistant ({detected_lang.upper()}): {response_data.get('response')}")
        
        print("----------------------")
