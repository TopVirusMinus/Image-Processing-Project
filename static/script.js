const uploadInput = document.querySelector('input[name="image-upload"]');
const uploadedImage = document.querySelector(".original-image");
const filteredImage = document.querySelector(".filtered-image");
const arrow = document.querySelector(".arrow");
const applyFilterButton = document.querySelector(".dropbtn");
const filterSelect = document.querySelector(".filter-select");
const kernelSizeSlider = document.querySelector("#kernel-size");
const kernelSizeValue = document.querySelector("#kernel-size-value");

uploadInput.addEventListener("change", function () {
  const file = this.files[0];
  const reader = new FileReader();
  reader.addEventListener("load", function () {
    uploadedImage.src = reader.result;
    uploadedImage.removeAttribute("hidden");

    filteredImage.src = reader.result;
    filteredImage.removeAttribute("hidden");

    arrow.removeAttribute("hidden");
  });
  reader.readAsDataURL(file);
});

applyFilterButton.addEventListener("click", async function () {
  const filterType = filterSelect.value;
  const kernelSize = kernelSizeSlider.value;
  kernelSizeValue.textContent = kernelSize;

  const response = await fetch("/process-image", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: `image_data=${encodeURIComponent(
      uploadedImage.src
    )}&filter_type=${filterType}&kernel_size=${kernelSize}`,
  });

  const result = await response.json();
  filteredImage.src = result.processed_image;
});

kernelSizeSlider.addEventListener("input", function () {
  const kernelSize = kernelSizeSlider.value;
  kernelSizeValue.textContent = kernelSize;
});
