import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any
from config.settings import Config
from config.database import MySQLDatabase, SQLiteDatabase

logger = logging.getLogger(__name__)


class AttendanceLogger:
    """Manages attendance time-in/time-out logic"""
    
    def __init__(self, mysql_db: MySQLDatabase, sqlite_db: SQLiteDatabase):
        self.mysql_db = mysql_db
        self.sqlite_db = sqlite_db
        self.last_scan_cache = {}  # Prevent duplicate scans
    
    def log_timein(self, worker_id: int) -> Dict[str, Any]:
        """
        Log worker time-in
        
        Returns:
            {
                'success': bool,
                'action': 'timein' | 'duplicate' | 'error',
                'message': str,
                'worker_info': dict
            }
        """
        today = date.today().isoformat()
        now = datetime.now()
        
        # Check duplicate scan cache
        cache_key = f"{worker_id}_{today}"
        if cache_key in self.last_scan_cache:
            last_scan = self.last_scan_cache[cache_key]
            if (now - last_scan).seconds < Config.DUPLICATE_TIMEOUT_SECONDS:
                return {
                    'success': False,
                    'action': 'duplicate',
                    'message': 'Already scanned recently'
                }
        
        # Check if time-in exists today
        if self.mysql_db.is_connected:
            existing = self.mysql_db.fetch_one("""
                SELECT attendance_id, time_out FROM attendance
                WHERE worker_id = %s AND attendance_date = %s
                AND is_archived = 0
            """, (worker_id, today))
            
            if existing:
                if existing['time_out'] is None:
                    # Already timed-in, offer time-out
                    return {
                        'success': False,
                        'action': 'already_in',
                        'message': 'Already timed in. Ready for time-out?',
                        'attendance_id': existing['attendance_id']
                    }
                else:
                    # Already completed
                    return {
                        'success': False,
                        'action': 'completed',
                        'message': 'Attendance already completed today'
                    }
        
        # Insert time-in
        time_in = now.strftime('%H:%M:%S')
        
        if self.mysql_db.is_connected:
            # Direct to MySQL
            query = """
                INSERT INTO attendance 
                (worker_id, attendance_date, time_in, status)
                VALUES (%s, %s, %s, 'present')
            """
            attendance_id = self.mysql_db.execute_query(query, (worker_id, today, time_in))
            
            if attendance_id:
                # Log activity
                self.mysql_db.execute_query("""
                    INSERT INTO activity_logs 
                    (user_id, action, table_name, record_id, description, ip_address)
                    VALUES (%s, 'clock_in', 'attendance', %s, 'Facial recognition time-in', 'raspberry_pi')
                """, (worker_id, attendance_id))
        else:
            # Buffer to SQLite
            attendance_id = self.sqlite_db.insert_attendance(
                worker_id, today, time_in=time_in
            )
        
        # Update cache
        self.last_scan_cache[cache_key] = now
        
        return {
            'success': True,
            'action': 'timein',
            'message': 'Time-in recorded successfully',
            'attendance_id': attendance_id,
            'time_in': time_in
        }
    
    def log_timeout(self, worker_id: int) -> Dict[str, Any]:
        """
        Log worker time-out
        
        Returns:
            {
                'success': bool,
                'action': 'timeout' | 'no_timein' | 'error',
                'message': str,
                'hours_worked': float
            }
        """
        today = date.today().isoformat()
        now = datetime.now()
        time_out = now.strftime('%H:%M:%S')
        
        # Find today's time-in
        if self.mysql_db.is_connected:
            record = self.mysql_db.fetch_one("""
                SELECT attendance_id, time_in FROM attendance
                WHERE worker_id = %s AND attendance_date = %s
                AND time_out IS NULL AND is_archived = 0
            """, (worker_id, today))
            
            if not record:
                return {
                    'success': False,
                    'action': 'no_timein',
                    'message': 'No time-in found for today'
                }
            
            # Calculate hours
            time_in_dt = datetime.strptime(record['time_in'], '%H:%M:%S')
            hours_worked = (now - time_in_dt).seconds / 3600
            
            # Update time-out
            self.mysql_db.execute_query("""
                UPDATE attendance 
                SET time_out = %s, hours_worked = %s, updated_at = NOW()
                WHERE attendance_id = %s
            """, (time_out, hours_worked, record['attendance_id']))
            
            # Log activity
            self.mysql_db.execute_query("""
                INSERT INTO activity_logs 
                (user_id, action, table_name, record_id, description, ip_address)
                VALUES (%s, 'clock_out', 'attendance', %s, 'Facial recognition time-out', 'raspberry_pi')
            """, (worker_id, record['attendance_id']))
        else:
            # Buffer to SQLite
            # Calculate approximate hours (assumes buffered time-in exists)
            hours_worked = 8.0  # Default estimate
            success = self.sqlite_db.update_timeout(
                worker_id, today, time_out, hours_worked
            )
            
            if not success:
                return {
                    'success': False,
                    'action': 'no_timein',
                    'message': 'No time-in found in buffer'
                }
        
        return {
            'success': True,
            'action': 'timeout',
            'message': 'Time-out recorded successfully',
            'time_out': time_out,
            'hours_worked': round(hours_worked, 2)
        }