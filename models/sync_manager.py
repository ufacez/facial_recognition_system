import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from config.settings import Config
from config.database import MySQLDatabase, SQLiteDatabase

logger = logging.getLogger(__name__)


class SyncManager:
    """Synchronizes offline buffer with central MySQL database"""
    
    def __init__(self, mysql_db: MySQLDatabase, sqlite_db: SQLiteDatabase):
        self.mysql_db = mysql_db
        self.sqlite_db = sqlite_db
        self.retry_count = {}
    
    def sync_all(self) -> Dict[str, int]:
        """
        Sync all pending records from SQLite to MySQL
        
        Returns:
            {'synced': int, 'failed': int, 'pending': int}
        """
        if not self.mysql_db.is_connected:
            if not self.mysql_db.connect():
                logger.warning("Cannot sync: MySQL unavailable")
                return {'synced': 0, 'failed': 0, 'pending': 0}
        
        pending = self.sqlite_db.get_pending_records()
        synced_count = 0
        failed_count = 0
        
        for record in pending:
            buffer_id = record['id']
            
            # Check retry limit
            retry_key = f"buffer_{buffer_id}"
            if retry_key in self.retry_count:
                if self.retry_count[retry_key] >= Config.MAX_RETRY_ATTEMPTS:
                    logger.error(f"Max retries reached for buffer ID {buffer_id}")
                    failed_count += 1
                    continue
            
            # Sync record
            success = self._sync_record(record)
            
            if success:
                self.sqlite_db.mark_synced(buffer_id)
                synced_count += 1
                
                # Clear retry count
                if retry_key in self.retry_count:
                    del self.retry_count[retry_key]
            else:
                failed_count += 1
                
                # Increment retry count
                self.retry_count[retry_key] = self.retry_count.get(retry_key, 0) + 1
        
        remaining = len(pending) - synced_count - failed_count
        
        logger.info(f"Sync complete: {synced_count} synced, {failed_count} failed, {remaining} pending")
        return {'synced': synced_count, 'failed': failed_count, 'pending': remaining}
    
    def _sync_record(self, record: Dict[str, Any]) -> bool:
        """Sync single attendance record to MySQL"""
        try:
            # Check if record already exists (prevent duplicates)
            existing = self.mysql_db.fetch_one("""
                SELECT attendance_id FROM attendance
                WHERE worker_id = %s AND attendance_date = %s
                AND is_archived = 0
            """, (record['worker_id'], record['attendance_date']))
            
            if existing:
                # Update existing record
                if record['time_out']:
                    self.mysql_db.execute_query("""
                        UPDATE attendance
                        SET time_out = %s, hours_worked = %s, updated_at = NOW()
                        WHERE attendance_id = %s AND time_out IS NULL
                    """, (record['time_out'], record['hours_worked'], existing['attendance_id']))
                
                logger.info(f"Updated existing attendance {existing['attendance_id']}")
            else:
                # Insert new record
                self.mysql_db.execute_query("""
                    INSERT INTO attendance
                    (worker_id, attendance_date, time_in, time_out, status, hours_worked)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    record['worker_id'],
                    record['attendance_date'],
                    record['time_in'],
                    record['time_out'],
                    record['status'],
                    record['hours_worked']
                ))
                
                logger.info(f"Inserted new attendance for worker {record['worker_id']}")
            
            return True
        except Exception as e:
            logger.error(f"Sync failed for buffer ID {record['id']}: {e}")
            return False