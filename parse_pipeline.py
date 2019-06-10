from parsing import parse_dicom_file, poly_to_mask, parse_contour_file
import matplotlib.pyplot as plt
import numpy as np 
import csv
import os

class DICOMParser():
	"""DICOMParser class which loads dicom/icounter files and pairs them according to 
	the link file and ids. It also converts the icounter files to boolean masks and saves
	the data as an easy to use X,Y numpy pair. 
	"""

	def __init__(self, data_dir, link_file):
		"""Initialize parser with the link_file indicating which directories
		correspond to each other 

		:param link_file: csv file containing link information
		:return: DICOMParser
		"""
		self.data_dir = data_dir
		self.link_file = link_file
		self.links = self.extract_links(link_file)	

	def extract_links(self, link_file):
		"""Extract the link information which maps contour directories 
		to corresponding dicom directories

		:param link_file: csv file containing link information
		:return: dictionary mapping directories
		"""
		result_links = {}
		with open(link_file, newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			next(reader) # Ignore the header line
			for row in reader: 
				patient_id = row[0]
				original_id = row[1]
				result_links[patient_id] = original_id
		return result_links

	def construct_icontour_dir(self, original_id):
		"""Construct icontour directory given the id

		:param original_id: original_id (e.g. "SC-HF-I-1")
		:return: the full icontour directory
		"""
		return self.data_dir + "/contourfiles/" + original_id + "/i-contours"

	def construct_dicom_dir(self, patient_id):
		"""Construct dicom directory given the id

		:param patient_id: patient_id (e.g. "SCD0000101")
		:return: the full dicom directory
		"""
		return self.data_dir + "/dicoms/" + patient_id
	
	def extract_id(self, filename):
		"""Extract the id from the given icontour filename

		:param filename: icontour filename (e.g. "IM-0001-0048-icontour-manual.txt")
		:return: the id (e.g. 48)
		"""
		return int(filename.split("-")[2])

	def construct_dicom_filename(self, id):
		"""Construct dicom filenmae given the id

		:param id: id (e.g. 123)
		:return: the dicom filename (e.g. 123.dcm)
		"""
		return str(id) + ".dcm"
		
	def parse_and_save(self):
		"""Main funtion to be called by users. This parses the dicom/icounter files,
		pairs them, and converts the icounter files to boolean masks. It then saves
		them in an X,Y numpy pair to be used later for training.

		:param id: id (e.g. 123)
		:return: the dicom filename (e.g. 123.dcm)
		"""
		X = []
		Y = []
		
		# Go through corresponding directories
		for patient_id, original_id in self.links.items():
			contour_dir = self.construct_icontour_dir(original_id)
			
			# List all contour files in the icontour directory
			for icontour_file in os.listdir(contour_dir):
				
				# Extract the id and construct the "x" and "y" file names (dicom/icontour files respectively)
				id = self.extract_id(icontour_file)
				x_file = self.construct_dicom_dir(patient_id) + \
					"/" + self.construct_dicom_filename(id)
				y_file = self.construct_icontour_dir(original_id) + \
					"/" + icontour_file
				
				# If we have a corresponding dicom image for the icontour label we
				# are looking at, generate a training (x,y) pair		
				if os.path.isfile(x_file):
					parsed_contour_file = parse_contour_file(y_file)
					parsed_dicom_file = parse_dicom_file(x_file)
					width, height = parsed_dicom_file.shape
					bool_mask = poly_to_mask(parsed_contour_file, width, height)
					plt.imshow(bool_mask, cmap="gray")
					plt.show()
					plt.imshow(parsed_dicom_file, cmap="gray")
					plt.show()
					X.append(parsed_dicom_file)
					Y.append(bool_mask)
		# Save data
		X = np.asarray(X)
		Y = np.asarray(Y)
		np.save("X", X)
		np.save("Y", Y)

class BatchIndexOutOfBoundsException(Exception):
	"""Custom exception to be raised when the batch generator 
	finishes passing through the entire dataset. 
	"""
	pass

class BatchGenerator():
	"""BatchGenertor class which loads the saved X,Y data, and batches it according 
	to the batch size provided. When the whole dataset has been looped through, the
	generator can be reset to provide for many epochs.
	"""

	def __init__(self, X_filename, Y_filename, batch_size=8):
		"""BatchGenertor class which loads the saved X,Y data, and batches it according 
		to the batch size provided. When the whole dataset has been looped through, the
		generator can be reset to provide for many epochs.

		:param X_filename: The filename containing the training samples 
		:param Y_filename: The filename containing the training labels 
		:param batch_size: The batch size to be used
		"""
		
		# Load saved data
		self.X = np.load(X_filename)
		self.Y = np.load(Y_filename)
		
		# Save batch size and number of batches
		self.batch_size = batch_size
		self.num_samples = self.X.shape[0]	
		self.num_batches = int(np.ceil(float(self.num_samples / batch_size)))
		self.batch_index = 0
		
		# Shuffle data to start to ensure random ordering
		self.shuffle_data()

	def reset_generator(self):
		"""Resets the generator to its original state so the dataset can be
		looped through multiple times. 
		"""
		self.batch_index = 0
		self.shuffle_data()

	def shuffle_data(self):
		"""Shuffles the data to ensure the dataset is looped through in 
		a random order. 
		"""
		p = np.random.permutation(self.num_samples)
		self.X = self.X[p]
		self.Y = self.Y[p]

	def get_batch(self):
		"""Gets an X,Y batch to be used in training. 

		:return: The X,Y numpy arries with size batch_size
		"""
		if self.batch_index < self.num_batches:
			print(self.batch_index*self.batch_size,(self.batch_index+1)*self.batch_size)
			X_batch = self.X[self.batch_index*self.batch_size:(self.batch_index+1)*self.batch_size] 
			Y_batch = self.Y[self.batch_index*self.batch_size:(self.batch_index+1)*self.batch_size]
			self.batch_index += 1
			return X_batch, Y_batch
		else:
			raise BatchIndexOutOfBoundsException("Epoch complete. Use reset_generator() to reset the generator.")
		

if __name__ == "__main__":
	# Example parser usage.
	parser = DICOMParser("final_data", "final_data/link.csv")
	parser.parse_and_save()

	# Example batch generator usage.
	batch_generator = BatchGenerator("X.npy", "Y.npy")
	num_epochs = 3
	for i in range(num_epochs):
		while True:
			try:
				X, Y = batch_generator.get_batch()
				# X, Y used to feed into training pipeline
			except BatchIndexOutOfBoundsException:
				batch_generator.reset_generator()
				# Computing per epoch metrics (accuracy, loss, etc.) can be done here.
				break
	
