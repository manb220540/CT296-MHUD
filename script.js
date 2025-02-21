async function predict() {
    const data = {
        parents: document.getElementById("parents").value,
        has_nurs: document.getElementById("has_nurs").value,
        form: document.getElementById("form").value,
        children: document.getElementById("children").value,
        housing: document.getElementById("housing").value,
        finance: document.getElementById("finance").value,
        social: document.getElementById("social").value,
        health: document.getElementById("health").value,
        model: document.getElementById("model").value // Send selected model
    };

    const response = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();
    // Kiểm tra nếu có lỗi từ server
    if (result.error) {
        document.getElementById("result").innerHTML = `<p style="color: red;">Lỗi: ${result.message}</p>`;
        return;
    }

    // Hiển thị kết quả dự đoán
    displayPrediction(result);
}
function displayPrediction(response) {
    let resultDiv = document.getElementById("result");
    let probabilities = response.probabilities;
    // Lấy phần tử select
    let selectElement = document.getElementById("model");

    // Lấy phần tử option được chọn
    let selectedOption = selectElement.options[selectElement.selectedIndex];

    // Lấy văn bản của phần tử option được chọn
    let selectedTextModel = selectedOption.text;

    let probText = "<h3>Kết quả dự đoán theo mô hình "+selectedTextModel+": <strong>" + response.predicted_label + "</strong></h3>";
    probText += "<h4>Tỉ lệ phần trăm:</h4><ul>";
    console.log(probabilities);
    // Khởi tạo mảng để lưu dữ liệu phần trăm dạng số cho biểu đồ
    let chartDataPoints = [];

    for (let label in probabilities) {
        // Chuyển đổi giá trị phần trăm sang số (float) và lưu vào mảng
        let percentage = parseFloat(probabilities[label]);
        chartDataPoints.push(percentage);

        probText += `<li><strong>${label}</strong>: ${percentage}%</li>`;
    }

    probText += "</ul>";
    resultDiv.innerHTML = probText;

    // Vẽ biểu đồ cột
    const ctx = document.getElementById("probabilities-chart").getContext("2d");
    const chartData = {
        labels: Object.keys(probabilities),
        datasets: [{
            label: 'Tỉ lệ phần trăm',
            // Sử dụng mảng dữ liệu số đã xử lý
            data: chartDataPoints,
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
        }]
    };

    const chartOptions = {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                // Đảm bảo trục Y hiển thị đúng phần trăm (từ 0 đến 100)
                max: 100 
            }
        }
    };

    // Tạo biểu đồ (nếu đã tồn tại, hãy huỷ biểu đồ cũ trước khi tạo mới)
    let existingChart = Chart.getChart("probabilities-chart"); // Lấy instance của chart cũ
    if (existingChart) {
        existingChart.destroy(); // Huỷ chart cũ
    }
    new Chart(ctx, {
        type: 'bar',
        data: chartData,
        options: chartOptions
    });
}
