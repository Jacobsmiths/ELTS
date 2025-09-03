# Improved Eye-Tracking Laser Turret System (ELTS)
import threading
import cv2
import mediapipe as mp
import busio
import board
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import csv
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
from contextlib import contextmanager
from GUI import GUI
from PID import PIDController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ServoConfig:
    """Configuration for servo parameters"""
    min_pulse: int = 500
    max_pulse: int = 2450
    frequency: int = 50
    min_angle: int = 0
    max_angle: int = 180
    center_angle: int = 90

@dataclass
class PIDConfig:
    """Configuration for PID parameters"""
    kp: float
    ki: float
    kd: float

@dataclass
class CameraConfig:
    """Configuration for camera parameters"""
    width: int = 640
    height: int = 480
    fps: int = 60
    buffer_size: int = 1

@dataclass
class TrackingData:
    """Data structure for tracking information"""
    timestamp: float
    x_coord: int
    y_coord: int
    servo_x_angle: float
    servo_y_angle: float

class EyeTracker:
    """Handles eye detection and coordinate calculation"""
    
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    
    def __init__(self, detection_confidence: float = 0.3, tracking_confidence: float = 0.2):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
    
    def detect_eyes(self, frame: cv2.Mat) -> Optional[Tuple[int, int]]:
        """Detect eyes in frame and return center coordinates"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if not results.multi_face_landmarks:
                return None
                
            mesh = results.multi_face_landmarks[0].landmark
            frame_h, frame_w, _ = frame.shape
            
            # Calculate eye centers
            left_coords = [(int(mesh[p].x * frame_w), int(mesh[p].y * frame_h)) for p in self.LEFT_IRIS]
            right_coords = [(int(mesh[p].x * frame_w), int(mesh[p].y * frame_h)) for p in self.RIGHT_IRIS]
            
            left_center = self._calculate_center(left_coords)
            right_center = self._calculate_center(right_coords)
            
            # Average of both eyes for face center
            eye_center_x = (left_center[0] + right_center[0]) // 2
            eye_center_y = (left_center[1] + right_center[1]) // 2
            
            return (eye_center_x, eye_center_y)
            
        except Exception as e:
            logger.error(f"Error in eye detection: {e}")
            return None
    
    def _calculate_center(self, coords: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate center point of coordinates"""
        x_coords, y_coords = zip(*coords)
        return (sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords))
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

class ServoController:
    """Handles servo control and positioning"""
    
    def __init__(self, config: ServoConfig, x_channel: int = 0, y_channel: int = 1):
        self.config = config
        self._initialize_hardware(x_channel, y_channel)
        self.center_servos()
    
    def _initialize_hardware(self, x_channel: int, y_channel: int):
        """Initialize I2C and servo hardware"""
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            pca = PCA9685(i2c)
            pca.frequency = self.config.frequency
            
            self.x_servo = servo.Servo(
                pca.channels[x_channel],
                min_pulse=self.config.min_pulse,
                max_pulse=self.config.max_pulse
            )
            self.y_servo = servo.Servo(
                pca.channels[y_channel],
                min_pulse=self.config.min_pulse,
                max_pulse=self.config.max_pulse
            )
            logger.info("Servo hardware initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize servo hardware: {e}")
            raise
    
    def move_to(self, x_angle: float, y_angle: float):
        """Move servos to specified angles with bounds checking"""
        x_angle = self._clamp_angle(x_angle)
        y_angle = self._clamp_angle(y_angle)
        
        self.x_servo.angle = x_angle
        self.y_servo.angle = y_angle
    
    def _clamp_angle(self, angle: float) -> float:
        """Clamp angle to valid servo range"""
        return max(min(angle, self.config.max_angle), self.config.min_angle)
    
    def center_servos(self):
        """Move servos to center position"""
        self.move_to(self.config.center_angle, self.config.center_angle)
        logger.info("Servos centered")
    
    @property
    def current_angles(self) -> Tuple[float, float]:
        """Get current servo angles"""
        return (self.x_servo.angle, self.y_servo.angle)

class DataLogger:
    """Handles CSV data logging"""
    
    def __init__(self, filename: str = 'tracking_data.csv'):
        self.filename = filename
        self.fieldnames = ['timestamp', 'x_coord', 'y_coord', 'servo_x_angle', 'servo_y_angle']
        self.csvfile = None
        self.writer = None
        self.start_time = None
    
    @contextmanager
    def logging_context(self):
        """Context manager for CSV logging"""
        try:
            self.csvfile = open(self.filename, 'w', newline='')
            self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames)
            self.writer.writeheader()
            self.start_time = time.time()
            logger.info(f"Data logging started: {self.filename}")
            yield self
        finally:
            if self.csvfile:
                self.csvfile.close()
                logger.info("Data logging stopped")
    
    def log_data(self, data: TrackingData):
        """Log tracking data to CSV"""
        if self.writer and self.start_time:
            self.writer.writerow({
                'timestamp': data.timestamp - self.start_time,
                'x_coord': data.x_coord,
                'y_coord': data.y_coord,
                'servo_x_angle': data.servo_x_angle,
                'servo_y_angle': data.servo_y_angle
            })
            self.csvfile.flush()

class CameraManager:
    """Handles camera initialization and configuration"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = None
    
    def initialize(self) -> Optional[cv2.VideoCapture]:
        """Initialize camera with fallback options"""
        for camera_index in range(3):  # Try indices 0, 1, 2
            try:
                logger.info(f"Attempting camera index {camera_index}")
                cap = cv2.VideoCapture(camera_index)
                
                # Configure camera
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                cap.set(cv2.CAP_PROP_FPS, self.config.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
                
                # Test camera
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    logger.info(f"Camera {camera_index} initialized successfully")
                    self.cap = cap
                    return cap
                else:
                    cap.release()
                    
            except Exception as e:
                logger.error(f"Camera {camera_index} failed: {e}")
                if cap:
                    cap.release()
        
        logger.error("No working camera found")
        return None
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.cap:
            self.cap.release()
            logger.info("Camera released")

class ELTS:
    """Enhanced Laser Tracking System main class"""
    
    def __init__(self):
        # Configuration
        self.servo_config = ServoConfig()
        self.camera_config = CameraConfig()
        self.x_pid_config = PIDConfig(kp=0.03, ki=0.002, kd=0.0)
        self.y_pid_config = PIDConfig(kp=0.02, ki=0.002, kd=0.0)
        
        # State variables
        self.quit_application = False
        self.tracking_enabled = False
        self.x_offset = 0
        self.y_offset = 0
        
        # Initialize components
        self._initialize_components()
        
        logger.info("ELTS system initialized")
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Hardware components
            self.servo_controller = ServoController(self.servo_config)
            self.camera_manager = CameraManager(self.camera_config)
            
            # Software components
            self.eye_tracker = EyeTracker()
            self.data_logger = DataLogger()
            
            # PID controllers
            center_x = self.camera_config.width // 2
            center_y = self.camera_config.height // 2
            
            self.x_pid = PIDController(
                self.x_pid_config.kp, 
                self.x_pid_config.ki, 
                self.x_pid_config.kd, 
                center_x
            )
            self.y_pid = PIDController(
                self.y_pid_config.kp, 
                self.y_pid_config.ki, 
                self.y_pid_config.kd, 
                center_y
            )
            
            self.last_time = time.perf_counter()
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def update_servo_positions(self, x_coord: int, y_coord: int):
        """Update servo positions based on eye coordinates"""
        try:
            current_time = time.perf_counter()
            dt = current_time - self.last_time
            self.last_time = current_time

            # Apply calibration offsets
            target_x = x_coord + self.x_offset
            target_y = y_coord + self.y_offset
            
            # Calculate PID outputs
            current_x, current_y = self.servo_controller.current_angles
            
            delta_x = -self.x_pid.compute(target_x, dt) + current_x
            delta_y = -self.y_pid.compute(target_y, dt) + current_y
            
            # Move servos
            self.servo_controller.move_to(delta_x, delta_y)
            
        except Exception as e:
            logger.error(f"Error updating servo positions: {e}")
    
    def draw_tracking_overlay(self, frame: cv2.Mat, eye_coords: Tuple[int, int]):
        """Draw tracking visualization on frame"""
        x_coord, y_coord = eye_coords
        center_x = self.camera_config.width // 2
        center_y = self.camera_config.height // 2
        servo_x, servo_y = self.servo_controller.current_angles
        
        # Draw target circle
        cv2.circle(frame, (x_coord, y_coord), 5, (255, 0, 0), 2)
        
        # Draw crosshairs at center
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 2)
        
        # Display servo angles
        cv2.putText(frame, f"Servo X: {servo_x:.1f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Servo Y: {servo_y:.1f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.tracking_enabled else (0, 0, 255), 2)
    
    def start(self):
        """Main tracking loop"""
        logger.info("Starting ELTS tracking system")
        
        # Initialize camera
        cap = self.camera_manager.initialize()
        if not cap:
            logger.error("Failed to initialize camera")
            return
        
        try:
            with self.data_logger.logging_context():
                while not self.quit_application:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Failed to read frame from camera")
                        break
                    
                    frame = cv2.flip(frame, 1)  # Mirror image
                    eye_coords = self.eye_tracker.detect_eyes(frame)
                    
                    if eye_coords:
                        self.draw_tracking_overlay(frame, eye_coords)
                        
                        if self.tracking_enabled:
                            # Update servo positions
                            self.update_servo_positions(*eye_coords)
                            
                            # Log data
                            servo_x, servo_y = self.servo_controller.current_angles
                            tracking_data = TrackingData(
                                timestamp=time.time(),
                                x_coord=eye_coords[0],
                                y_coord=eye_coords[1],
                                servo_x_angle=servo_x,
                                servo_y_angle=servo_y
                            )
                            self.data_logger.log_data(tracking_data)
                    else:
                        cv2.putText(frame, "NO FACE DETECTED", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Display frame
                    cv2.imshow("Eye Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in main tracking loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up all resources"""
        logger.info("Cleaning up ELTS system")
        self.camera_manager.cleanup()
        self.eye_tracker.cleanup()
        cv2.destroyAllWindows()
    
    # Calibration methods
    def move_up(self):
        self.y_offset -= 5
        logger.info(f"Move up - Y offset: {self.y_offset}")
    
    def move_down(self):
        self.y_offset += 5
        logger.info(f"Move down - Y offset: {self.y_offset}")
    
    def move_right(self):
        self.x_offset += 5
        logger.info(f"Move right - X offset: {self.x_offset}")
    
    def move_left(self):
        self.x_offset -= 5
        logger.info(f"Move left - X offset: {self.x_offset}")
    
    def clear_offsets(self):
        self.x_offset = 0
        self.y_offset = 0
        logger.info("Offsets cleared")
    
    def start_tracking(self):
        self.tracking_enabled = True
        logger.info("Tracking started")
    
    def stop_tracking(self):
        self.tracking_enabled = False
        logger.info("Tracking stopped")
    
    def center_servos(self):
        self.servo_controller.center_servos()

def main():
    """Main entry point"""
    try:
        elts = ELTS()
        
        # Start tracking in separate thread
        tracking_thread = threading.Thread(target=elts.start, daemon=True)
        tracking_thread.start()
        
        # Initialize GUI in main thread
        gui = GUI(
            startCommand=elts.start_tracking,
            stopCommand=elts.stop_tracking,
            centerServosCommand=elts.center_servos,
            resetOffsetsCommand=elts.clear_offsets,
            upCommand=elts.move_up,
            downCommand=elts.move_down,
            rightCommand=elts.move_right,
            leftCommand=elts.move_left
        )
        gui.start()
        
        # Signal shutdown when GUI closes
        elts.quit_application = True
        
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        print("Program ended")

if __name__ == "__main__":
    main()