<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');

try {
    $pdo = new PDO('mysql:host=localhost;dbname=motor_control', 'root', '', [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION
    ]);

    $date = isset($_GET['date']) ? $_GET['date'] : null;
    $query = "
        SELECT timestamp, weight, count, speed, temperature, motor_status 
        FROM sensor_data 
        WHERE timestamp >= NOW() - INTERVAL 24 HOUR
        ORDER BY timestamp
    ";
    if ($date) {
        $query = "
            SELECT timestamp, weight, count, speed, temperature, motor_status 
            FROM sensor_data 
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp
        ";
    }
    $stmt = $pdo->prepare($query);
    if ($date) {
        $stmt->execute([$date]);
    } else {
        $stmt->execute();
    }
    $data = $stmt->fetchAll(PDO::FETCH_ASSOC);
    echo json_encode($data);
} catch (PDOException $e) {
    echo json_encode(['error' => 'Database error: ' . $e->getMessage()]);
}
?>
